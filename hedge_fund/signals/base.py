"""Alpha models — the components that form views on what to hold.

An *alpha model* (Rishi Narang's term, *Inside the Black Box*) is anything
that produces a forecast / view on an asset. It's the "edge" component of a
quant fund. In v2, both quant signals (PEAD, regime) and LLM investor agents
(Buffett, Druckenmiller) are alpha models — they all implement this interface
and produce a `Signal` (a conviction in [-1, +1] + reasoning).

    AlphaModel (ABC)
      ├─ QuantModel   — pure Python math (this file)
      └─ LLMAgent     — LLM reasons over features (added in Week 5)

The alpha model only forms a *view*. It does NOT decide position mechanics
(timing, sizing, holding period) — that's the job of portfolio construction
and execution. This separation (views vs positions) is deliberate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date as _date
from datetime import timedelta

import numpy as np
import pandas as pd

from hedge_fund.data.protocol import DataClient
from hedge_fund.models import Signal


class AlphaModel(ABC):
    """Abstract base for all alpha models. Forms a view, returns a Signal."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier (e.g. 'pead', 'buffett')."""
        ...

    @abstractmethod
    def predict(
        self,
        ticker: str,
        date: str,
        data_client: DataClient,
    ) -> Signal:
        """Form a point-in-time view on *ticker* as of *date*.

        MUST be point-in-time: only use data with date <= *date* (no
        lookahead). Return a Signal with conviction in [-1, +1] — use
        0.0 to express "no view" (abstain).
        """
        ...


class _CloseCache:
    """Per-ticker daily closes, merged across the windows asked for so far.

    A price-based model is called once per ticker per trading day during a
    backtest, and each call wants a long trailing window — refetching that
    window every day would be hundreds of redundant requests per ticker. So
    bars are merged into one per-ticker map and the covered date range is
    tracked; a request inside the covered range is served from memory, and
    one outside it refetches the union of the two windows.

    Serving from the cache stays point-in-time: every read filters bars to
    `date <= end`, so a window cached for a later date can never leak a
    future close into an earlier call.
    """

    def __init__(self) -> None:
        self._bars: dict[str, dict[str, float]] = {}
        self._covered: dict[str, tuple[str, str]] = {}

    def closes(
        self,
        ticker: str,
        start: str,
        end: str,
        data_client: DataClient,
    ) -> pd.Series:
        """Closes for *ticker* with start <= date <= end, ascending by date."""
        covered = self._covered.get(ticker)
        if covered is None or start < covered[0] or end > covered[1]:
            fetch_start = min(start, covered[0]) if covered else start
            fetch_end = max(end, covered[1]) if covered else end
            bars = self._bars.setdefault(ticker, {})
            for price in data_client.get_prices(ticker, fetch_start, fetch_end):
                bars[price.time[:10]] = float(price.close)
            self._covered[ticker] = (fetch_start, fetch_end)

        window = {
            day: close
            for day, close in self._bars.get(ticker, {}).items()
            if start <= day <= end
        }
        return pd.Series(window, dtype=float).sort_index()


class QuantModel(AlphaModel):
    """Base for pure-math alpha models (no LLM).

    Houses shared numeric helpers. Subclass this for quant signals like
    PEAD or regime detection.
    """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        """Convert to float, returning *default* for NaN / None / errors."""
        if value is None:
            return default
        try:
            f = float(value)
            return default if (np.isnan(f) or np.isinf(f)) else f
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _percentile_rank(value: float, values: list[float]) -> float:
        """Return the percentile rank (0-100) of *value* within *values*."""
        if not values:
            return 50.0
        below = sum(1 for v in values if v < value)
        return (below / len(values)) * 100.0

    @staticmethod
    def _normalize_to_signal(raw: float, low: float = -1.0, high: float = 1.0) -> float:
        """Clamp *raw* into [low, high]."""
        return max(low, min(high, raw))

    @staticmethod
    def _sigmoid(x: float, scale: float = 5.0) -> float:
        """Map an unbounded value into (-1, +1) via scaled tanh."""
        return float(np.tanh(x * scale))

    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = 14) -> float:
        """Compute the latest RSI value for a price series."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        latest = rsi.iloc[-1]
        if pd.isna(latest):
            return 50.0
        return float(latest)

    # ------------------------------------------------------------------
    # Price history
    # ------------------------------------------------------------------

    @property
    def _close_cache(self) -> _CloseCache:
        """Lazily-created per-instance price cache.

        A property rather than an ``__init__`` attribute so subclasses —
        which each define their own constructor — never have to remember to
        call ``super().__init__()`` to get caching.
        """
        cache = getattr(self, "_close_cache_store", None)
        if cache is None:
            cache = _CloseCache()
            self._close_cache_store = cache
        return cache

    def _closes(
        self,
        ticker: str,
        date: str,
        data_client: DataClient,
        trading_days: int,
    ) -> pd.Series:
        """The last *trading_days* closes on or before *date*, ascending.

        Point-in-time by construction: nothing after *date* is ever
        returned. The fetch window is padded to calendar days (markets open
        ~5 days in 7, plus holidays), then trimmed to the tail.
        """
        calendar_days = int(trading_days * 1.5) + 10
        start = (_date.fromisoformat(date) - timedelta(days=calendar_days)).isoformat()
        series = self._close_cache.closes(ticker, start, date, data_client)
        return series.iloc[-trading_days:]
