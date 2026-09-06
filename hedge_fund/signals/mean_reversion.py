"""Mean-reversion alpha model — short-horizon reversal.

The mirror image of momentum. Over days-to-weeks, price moves that overshoot
tend to snap back: liquidity demand, forced selling, and index-flow pressure
push a name away from its own recent average, and the pressure fades before
the fundamentals change. So the view here is *contrarian* — a stretch above
the moving average is a short, a stretch below is a long.

Two knobs keep it from fighting a genuine trend:

- **A deadband.** Inside `entry_z` standard deviations of the moving average
  the model votes a real 0.0 ("priced fairly right now"), not an abstention.
- **An RSI confirmation.** A name can sit far below its average because it is
  being repriced, not because it overshot. Requiring RSI to be genuinely
  oversold (or overbought) before fading the move is the textbook second
  opinion, and it is a strict one: on random walks it stands the model down on
  roughly six in ten of the stretches the z-score alone would have traded.

Pure math over closes — no LLM key needed to backtest it.
"""

from __future__ import annotations

import numpy as np

from hedge_fund.data.protocol import DataClient
from hedge_fund.models import Signal
from hedge_fund.signals.base import QuantModel


class MeanReversionModel(QuantModel):
    """Short what is stretched above its moving average, long what is below.

    `predict(ticker, date)` z-scores the latest close against its
    `window`-day moving average and standard deviation, then returns the
    *negated* z-score squashed into [-1, +1].

    Three outcomes, deliberately distinct:

    - **A view** — |z| >= `entry_z` and RSI confirms: a signed conviction.
    - **A real neutral (0.0)** — inside the deadband, or RSI does not confirm.
      The model looked and has nothing to say today; it dilutes the blend,
      the way PEAD does outside its event window.
    - **An abstention** — not enough history, or a flat/degenerate window.
      Excluded from the blend entirely.
    """

    def __init__(
        self,
        *,
        window: int = 21,
        entry_z: float = 1.0,
        scale: float = 0.5,
        rsi_period: int = 14,
        rsi_filter: bool = True,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
    ) -> None:
        if window < 3:
            raise ValueError("window must be at least 3")
        if entry_z < 0:
            raise ValueError("entry_z cannot be negative")
        if not 0 < rsi_oversold < rsi_overbought < 100:
            raise ValueError("need 0 < rsi_oversold < rsi_overbought < 100")
        self._window = window
        self._entry_z = entry_z
        self._scale = scale
        self._rsi_period = rsi_period
        self._rsi_filter = rsi_filter
        self._rsi_oversold = rsi_oversold
        self._rsi_overbought = rsi_overbought

    @property
    def name(self) -> str:
        return "mean_reversion"

    # RSI needs `rsi_period` differences on top of the z-score window.
    @property
    def _required_bars(self) -> int:
        return max(self._window, self._rsi_period + 1)

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        closes = self._closes(ticker, date, data_client, self._required_bars)
        if len(closes) < self._required_bars:
            return self._abstain(
                ticker, date,
                f"only {len(closes)} closes on or before {date}; "
                f"{self._required_bars} needed",
            )

        window = closes.iloc[-self._window:]
        values = window.to_numpy(dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        if not np.isfinite(std) or std <= 1e-9:
            return self._abstain(ticker, date, "price window is flat — no dispersion to z-score")

        last = float(values[-1])
        z = (last - mean) / std
        rsi = self._compute_rsi(closes, period=self._rsi_period)

        components = {"z_score": z, "moving_average": mean, "close": last, "rsi": rsi}

        if abs(z) < self._entry_z:
            return self._neutral(
                ticker, date, components,
                f"within {self._entry_z:.1f}σ of its {self._window}d average "
                f"(z={z:+.2f}) — priced fairly, no reversal to trade",
            )

        # Contrarian: stretched high -> short, stretched low -> long.
        value = self._normalize_to_signal(self._sigmoid(-z, self._scale))

        if self._rsi_filter and not self._rsi_confirms(value, rsi):
            threshold = self._rsi_oversold if value > 0 else self._rsi_overbought
            return self._neutral(
                ticker, date, components,
                f"{abs(z):.2f}σ {'above' if z > 0 else 'below'} its {self._window}d "
                f"average, but RSI {rsi:.0f} is not "
                f"{'oversold' if value > 0 else 'overbought'} "
                f"({threshold:.0f}) — stretched is not the same as exhausted",
            )

        side = "stretched above" if z > 0 else "stretched below"
        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=value,
            reasoning=(
                f"{last:.2f} is {abs(z):.2f}σ {side} its {self._window}d average "
                f"of {mean:.2f} (RSI {rsi:.0f}) — fading the move"
            ),
            components=components,
            metadata={
                "window": self._window,
                "entry_z": self._entry_z,
                "rsi_filter": self._rsi_filter,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rsi_confirms(self, value: float, rsi: float) -> bool:
        """A long needs a genuinely oversold RSI; a short, a genuinely overbought one."""
        if value > 0:
            return rsi <= self._rsi_oversold
        return rsi >= self._rsi_overbought

    def _neutral(
        self,
        ticker: str,
        date: str,
        components: dict[str, float],
        reasoning: str,
    ) -> Signal:
        """A real neutral vote — the model looked and has no edge today."""
        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=0.0,
            reasoning=reasoning,
            components=components,
        )

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
