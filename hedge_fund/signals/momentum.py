"""Momentum alpha model — risk-adjusted time-series momentum (12-1).

The oldest documented cross-sectional anomaly: names that went up over the
past year keep going up, names that went down keep going down (Jegadeesh &
Titman, 1993). The formation window deliberately *skips* the most recent
month, because at one-month horizons the effect reverses — that short-term
reversal is a separate edge, and `MeanReversionModel` harvests it.

The raw formation return is divided by the window's realized volatility
before it becomes a conviction, so a 40% run in a quiet name outranks the
same 40% in a name that swings 40% a quarter. That is the difference between
"it went up" and "it went up more than its own noise explains".

Pure math over closes, so it backtests without an LLM key. Like every alpha
model it only forms a *view*; sizing and timing belong to portfolio
construction and execution.
"""

from __future__ import annotations

import numpy as np

from hedge_fund.data.protocol import DataClient
from hedge_fund.models import Signal
from hedge_fund.signals.base import QuantModel

_TRADING_DAYS_PER_YEAR = 252


class MomentumModel(QuantModel):
    """Long past winners, short past losers, scaled by realized volatility.

    `predict(ticker, date)` measures the return from `lookback_days +
    skip_days` ago to `skip_days` ago, divides it by the annualized
    volatility over that same window, and squashes the ratio into
    [-1, +1].

    Abstains (conviction 0.0, `metadata.abstained`) when the ticker has too
    little price history to form the full window, or when its volatility is
    zero — "no opinion", which portfolio construction excludes from the
    blend rather than counting as a neutral vote.
    """

    def __init__(
        self,
        *,
        lookback_days: int = 252,
        skip_days: int = 21,
        scale: float = 0.5,
    ) -> None:
        if lookback_days < 2:
            raise ValueError("lookback_days must be at least 2")
        if skip_days < 0:
            raise ValueError("skip_days cannot be negative")
        self._lookback_days = lookback_days
        self._skip_days = skip_days
        self._scale = scale

    @property
    def name(self) -> str:
        return "momentum"

    # Bars needed to measure the formation window end to end.
    @property
    def _required_bars(self) -> int:
        return self._lookback_days + self._skip_days + 1

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        closes = self._closes(ticker, date, data_client, self._required_bars)
        if len(closes) < self._required_bars:
            return self._abstain(
                ticker, date,
                f"only {len(closes)} closes on or before {date}; "
                f"{self._required_bars} needed for a "
                f"{self._lookback_days}-day window skipping {self._skip_days}",
            )

        # Formation window: [-required, -1-skip]. The last `skip_days` bars
        # are held out so the freshest (mean-reverting) move never drives it.
        window = closes.iloc[: len(closes) - self._skip_days]
        start_price = float(window.iloc[0])
        end_price = float(window.iloc[-1])
        if start_price <= 0:
            return self._abstain(ticker, date, "non-positive price in the formation window")

        total_return = end_price / start_price - 1.0

        daily = np.diff(np.log(window.to_numpy()))
        vol = float(np.std(daily, ddof=1)) * np.sqrt(_TRADING_DAYS_PER_YEAR)
        if not np.isfinite(vol) or vol <= 1e-9:
            return self._abstain(ticker, date, "formation window has no measurable volatility")

        risk_adjusted = total_return / vol
        value = self._normalize_to_signal(self._sigmoid(risk_adjusted, self._scale))

        if abs(risk_adjusted) < 0.1:
            direction = "flat"
        else:
            direction = "winner" if total_return > 0 else "loser"
        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=value,
            reasoning=(
                f"{self._lookback_days}d momentum (skipping the last "
                f"{self._skip_days}d): {total_return:+.1%} on {vol:.1%} "
                f"annualized vol — a {direction} at {risk_adjusted:+.2f}x its own noise "
                f"({window.index[0]} → {window.index[-1]})"
            ),
            components={
                "total_return": total_return,
                "annualized_vol": vol,
                "risk_adjusted": risk_adjusted,
            },
            metadata={
                "window_start": str(window.index[0]),
                "window_end": str(window.index[-1]),
                "lookback_days": self._lookback_days,
                "skip_days": self._skip_days,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _abstain(self, ticker: str, date: str, reason: str) -> Signal:
        """No view — excluded from the blend, not counted as a neutral vote."""
        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=0.0,
            reasoning=f"no view: {reason}",
            metadata={"abstained": True, "reason": reason},
        )
