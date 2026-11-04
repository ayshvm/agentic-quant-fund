"""Deterministic transaction-cost models for simulated execution.

The Indian equity market has several statutory and exchange costs that are
not represented by a single commission-bps knob.  This module keeps those
costs explicit and configurable so a backtest can be audited and updated as
rates or a broker's schedule changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from hedge_fund.brokers.models import Order


@dataclass(frozen=True)
class IndianEquityCostModel:
    """NSE/BSE-style delivery-equity cost model.

    Rates are expressed as decimal fractions (0.001 = 0.10%).  The defaults
    are a transparent research baseline, not a broker quote; pass explicit
    values when reproducing a broker's current tariff.  DP charge is modeled
    per sell order, which is useful for conservative portfolio backtests.
    """

    brokerage_bps: float = 0.0
    stt_rate: float = 0.001
    exchange_transaction_rate: float = 0.0000325
    sebi_turnover_rate: float = 0.000001
    stamp_duty_rate: float = 0.00015
    gst_rate: float = 0.18
    dp_charge_per_sell: float = 15.93

    def __post_init__(self) -> None:
        rates = (
            self.brokerage_bps,
            self.stt_rate,
            self.exchange_transaction_rate,
            self.sebi_turnover_rate,
            self.stamp_duty_rate,
            self.gst_rate,
            self.dp_charge_per_sell,
        )
        if any(rate < 0 for rate in rates):
            raise ValueError("Indian transaction-cost rates cannot be negative")

    def calculate(self, order: Order) -> float:
        """Return total charges for one completely filled delivery order."""
        turnover = order.quantity * order.price
        brokerage = turnover * self.brokerage_bps / 10_000
        stt = turnover * self.stt_rate
        exchange = turnover * self.exchange_transaction_rate
        sebi = turnover * self.sebi_turnover_rate
        stamp = turnover * self.stamp_duty_rate if order.side == "buy" else 0.0
        gst = (brokerage + exchange) * self.gst_rate
        dp = self.dp_charge_per_sell if order.side == "sell" else 0.0
        return brokerage + stt + exchange + sebi + stamp + gst + dp
