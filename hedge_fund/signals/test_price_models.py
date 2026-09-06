"""Tests for the price-based quant models (momentum, mean reversion)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from hedge_fund.data.models import Price
from hedge_fund.signals import MeanReversionModel, MomentumModel
from hedge_fund.signals.base import QuantModel


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakePriceClient:
    """Serves a canned close series on consecutive calendar days.

    Records every window it was asked for, so the cache can be tested.
    """

    def __init__(self, closes: list[float], end: str = "2025-06-30"):
        last = date.fromisoformat(end)
        self.bars = [
            Price(
                open=c, close=c, high=c, low=c, volume=1000,
                time=f"{(last - timedelta(days=len(closes) - 1 - i)).isoformat()}T00:00:00Z",
            )
            for i, c in enumerate(closes)
        ]
        self.calls: list[tuple[str, str, str]] = []

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        self.calls.append((ticker, start_date, end_date))
        return [b for b in self.bars if start_date <= b.time[:10] <= end_date]


def _ramp(n: int, start: float = 100.0, step: float = 0.5) -> list[float]:
    """A steadily rising series with a little wobble so vol is non-zero."""
    return [start + step * i + (0.3 if i % 2 else -0.3) for i in range(n)]


def _zigzag_down(n: int = 40, down: float = 3.0, up: float = 2.0) -> list[float]:
    """A grinding decline: down more than it goes up, but up often enough
    that RSI never reaches oversold. Stretched, not exhausted."""
    closes = [100.0]
    for i in range(n - 1):
        closes.append(closes[-1] - (down if i % 2 == 0 else -up))
    return closes


# ---------------------------------------------------------------------------
# Shared price helpers on QuantModel
# ---------------------------------------------------------------------------

class TestCloseHistory:
    def test_returns_the_tail_ascending(self):
        client = FakePriceClient([1.0, 2.0, 3.0, 4.0, 5.0])
        closes = MomentumModel()._closes("TEST", "2025-06-30", client, 3)
        assert list(closes) == [3.0, 4.0, 5.0]
        assert list(closes.index) == ["2025-06-28", "2025-06-29", "2025-06-30"]

    def test_never_returns_bars_after_the_as_of_date(self):
        client = FakePriceClient([1.0, 2.0, 3.0, 4.0, 5.0])
        model = MomentumModel()
        model._closes("TEST", "2025-06-30", client, 5)   # caches through 06-30
        closes = model._closes("TEST", "2025-06-28", client, 5)
        assert list(closes) == [1.0, 2.0, 3.0]

    def test_second_call_inside_the_window_is_served_from_cache(self):
        client = FakePriceClient(_ramp(60))
        model = MomentumModel()
        model._closes("TEST", "2025-06-30", client, 40)
        assert len(client.calls) == 1
        model._closes("TEST", "2025-06-29", client, 20)
        assert len(client.calls) == 1, "cached window should have covered the request"

    def test_cache_is_per_instance(self):
        client = FakePriceClient(_ramp(30))
        MomentumModel()._closes("TEST", "2025-06-30", client, 10)
        MomentumModel()._closes("TEST", "2025-06-30", client, 10)
        assert len(client.calls) == 2


# ---------------------------------------------------------------------------
# MomentumModel
# ---------------------------------------------------------------------------

class TestMomentum:
    def test_name_and_registry(self):
        assert MomentumModel().name == "momentum"
        assert isinstance(MomentumModel(), QuantModel)

    def test_uptrend_is_bullish(self):
        model = MomentumModel(lookback_days=60, skip_days=5)
        client = FakePriceClient(_ramp(120))
        sig = model.predict("TEST", "2025-06-30", client)
        assert sig.value > 0
        assert sig.components["total_return"] > 0
        assert not sig.metadata.get("abstained")

    def test_downtrend_is_bearish(self):
        model = MomentumModel(lookback_days=60, skip_days=5)
        client = FakePriceClient(list(reversed(_ramp(120))))
        sig = model.predict("TEST", "2025-06-30", client)
        assert sig.value < 0
        assert sig.components["total_return"] < 0

    def test_conviction_stays_in_range(self):
        model = MomentumModel(lookback_days=60, skip_days=5)
        client = FakePriceClient(_ramp(120, step=8.0))
        sig = model.predict("TEST", "2025-06-30", client)
        assert -1.0 <= sig.value <= 1.0

    def test_skip_window_is_excluded_from_the_formation_return(self):
        """A spike inside the skip window must not move the view."""
        base = _ramp(120)
        model = MomentumModel(lookback_days=60, skip_days=5)
        quiet = model.predict("TEST", "2025-06-30", FakePriceClient(base))

        spiked = list(base)
        spiked[-3] *= 3.0                      # inside the 5-day skip window
        loud = MomentumModel(lookback_days=60, skip_days=5).predict(
            "TEST", "2025-06-30", FakePriceClient(spiked),
        )
        assert loud.value == pytest.approx(quiet.value)

    def test_risk_adjustment_prefers_the_calmer_winner(self):
        """Same start, same finish, more noise in between -> weaker conviction."""
        calm = [100.0 + i for i in range(120)]
        choppy = list(calm)
        for i in range(1, len(choppy) - 1):          # endpoints untouched
            choppy[i] += 12.0 if i % 2 else -12.0

        model = MomentumModel(lookback_days=119, skip_days=0)
        calm_sig = model.predict("TEST", "2025-06-30", FakePriceClient(calm))
        choppy_sig = MomentumModel(lookback_days=119, skip_days=0).predict(
            "TEST", "2025-06-30", FakePriceClient(choppy),
        )
        assert calm_sig.components["total_return"] == pytest.approx(
            choppy_sig.components["total_return"]
        )
        assert calm_sig.value > choppy_sig.value

    def test_abstains_without_enough_history(self):
        model = MomentumModel(lookback_days=60, skip_days=5)
        sig = model.predict("TEST", "2025-06-30", FakePriceClient(_ramp(20)))
        assert sig.value == 0.0
        assert sig.metadata["abstained"] is True

    def test_abstains_on_a_flat_series(self):
        model = MomentumModel(lookback_days=60, skip_days=5)
        sig = model.predict("TEST", "2025-06-30", FakePriceClient([100.0] * 120))
        assert sig.metadata["abstained"] is True

    def test_abstains_when_the_ticker_has_no_prices(self):
        model = MomentumModel(lookback_days=60, skip_days=5)
        sig = model.predict("TEST", "2025-06-30", FakePriceClient([]))
        assert sig.metadata["abstained"] is True

    def test_rejects_nonsense_parameters(self):
        with pytest.raises(ValueError):
            MomentumModel(lookback_days=1)
        with pytest.raises(ValueError):
            MomentumModel(skip_days=-1)


# ---------------------------------------------------------------------------
# MeanReversionModel
# ---------------------------------------------------------------------------

class TestMeanReversion:
    def test_name_and_registry(self):
        assert MeanReversionModel().name == "mean_reversion"
        assert isinstance(MeanReversionModel(), QuantModel)

    def test_spike_above_the_average_is_bearish(self):
        closes = [100.0 + (1.0 if i % 2 else -1.0) for i in range(40)]
        closes[-1] = 130.0
        sig = MeanReversionModel(rsi_filter=False).predict(
            "TEST", "2025-06-30", FakePriceClient(closes),
        )
        assert sig.value < 0
        assert sig.components["z_score"] > 0

    def test_slump_below_the_average_is_bullish(self):
        closes = [100.0 + (1.0 if i % 2 else -1.0) for i in range(40)]
        closes[-1] = 70.0
        sig = MeanReversionModel(rsi_filter=False).predict(
            "TEST", "2025-06-30", FakePriceClient(closes),
        )
        assert sig.value > 0
        assert sig.components["z_score"] < 0

    def test_deadband_votes_a_real_neutral_not_an_abstention(self):
        closes = [100.0 + (1.0 if i % 2 else -1.0) for i in range(40)]
        sig = MeanReversionModel(entry_z=2.0).predict(
            "TEST", "2025-06-30", FakePriceClient(closes),
        )
        assert sig.value == 0.0
        assert "abstained" not in sig.metadata
        assert "z=" in sig.reasoning

    def test_rsi_confirmation_blocks_a_stretch_that_is_not_exhausted(self):
        """A grinding zigzag decline: 1.8σ below the average, RSI only 40."""
        closes = _zigzag_down()
        blocked = MeanReversionModel(rsi_filter=True).predict(
            "TEST", "2025-06-30", FakePriceClient(closes),
        )
        assert blocked.components["z_score"] < -1.0
        assert blocked.components["rsi"] > 30.0
        assert blocked.value == 0.0
        assert "abstained" not in blocked.metadata
        assert "not oversold" in blocked.reasoning

        unfiltered = MeanReversionModel(rsi_filter=False).predict(
            "TEST", "2025-06-30", FakePriceClient(closes),
        )
        assert unfiltered.value > 0, "the z-score alone would have bought it"

    def test_rsi_confirmation_lets_an_exhausted_stretch_through(self):
        """A one-way slide: oversold RSI confirms the long."""
        closes = [100.0 - 1.5 * i for i in range(40)]
        sig = MeanReversionModel(rsi_filter=True).predict(
            "TEST", "2025-06-30", FakePriceClient(closes),
        )
        assert sig.components["rsi"] <= 30.0
        assert sig.value > 0

    def test_rsi_thresholds_are_configurable(self):
        closes = _zigzag_down()
        strict = MeanReversionModel(rsi_filter=True).predict(
            "TEST", "2025-06-30", FakePriceClient(closes),
        )
        loose = MeanReversionModel(rsi_filter=True, rsi_oversold=45.0).predict(
            "TEST", "2025-06-30", FakePriceClient(closes),
        )
        assert strict.value == 0.0
        assert loose.value > 0

    def test_conviction_stays_in_range(self):
        closes = [100.0 + (1.0 if i % 2 else -1.0) for i in range(40)]
        closes[-1] = 1000.0
        sig = MeanReversionModel(rsi_filter=False).predict(
            "TEST", "2025-06-30", FakePriceClient(closes),
        )
        assert -1.0 <= sig.value <= 1.0

    def test_abstains_without_enough_history(self):
        sig = MeanReversionModel(window=21).predict(
            "TEST", "2025-06-30", FakePriceClient([100.0, 101.0]),
        )
        assert sig.value == 0.0
        assert sig.metadata["abstained"] is True

    def test_abstains_on_a_flat_window(self):
        sig = MeanReversionModel().predict(
            "TEST", "2025-06-30", FakePriceClient([100.0] * 40),
        )
        assert sig.metadata["abstained"] is True

    def test_rejects_nonsense_parameters(self):
        with pytest.raises(ValueError):
            MeanReversionModel(window=2)
        with pytest.raises(ValueError):
            MeanReversionModel(entry_z=-0.5)
        with pytest.raises(ValueError):
            MeanReversionModel(rsi_oversold=80.0, rsi_overbought=20.0)
