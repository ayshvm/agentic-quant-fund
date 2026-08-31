"""Risk limits — hard caps the analysts cannot override.

"Conviction requests, risk disposes": portfolio construction proposes target
weights, and this stage clamps them against the fund's limits. Everything
here is deterministic arithmetic — the LLM's influence over the book ends at
the Signal, and no clamp is ever negotiable.

Exposure removed by a clamp is NOT redistributed to other names; it stays in
cash. Redistributing would let the risk stage *increase* positions, which
inverts its job.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RiskLimits(BaseModel):
    """The fund's hard limits, set in its FundSpec."""

    model_config = ConfigDict(extra="forbid")

    max_position_pct: float = Field(
        gt=0, le=1.0, description="max |weight| per ticker, as a fraction of equity"
    )
    max_gross_exposure: float = Field(
        gt=0, description="max sum of |weights| across the book (1.0 = unlevered)"
    )
    max_net_exposure: float | None = Field(
        default=None, ge=0, le=1.0,
        description="optional cap on |sum(weights)|; excess stays in cash",
    )
    max_short_exposure: float | None = Field(
        default=None, ge=0,
        description="optional cap on the sum of absolute short weights",
    )


class ClampEvent(BaseModel):
    """One limit firing — recorded so every clamp is explainable."""

    limit: Literal[
        "max_position_pct", "max_gross_exposure", "max_net_exposure",
        "max_short_exposure",
    ]
    ticker: str | None = None  # None for the portfolio-level gross clamp
    before: float
    after: float


class RiskResult(BaseModel):
    """Clamped weights plus the audit trail of every limit that fired."""

    weights: dict[str, float]
    clamps: list[ClampEvent]


def apply_limits(weights: dict[str, float], limits: RiskLimits) -> RiskResult:
    """Clamp target weights against the fund's hard limits.

    Order matters and makes the pair idempotent:
    1. Per-ticker cap: any |weight| above max_position_pct is clamped to the
       cap, preserving sign. One ClampEvent per clamped ticker.
    2. Gross cap: if the summed |weights| still exceed max_gross_exposure,
       every weight is scaled down proportionally. Scaling only shrinks, so
       it can never re-violate the per-ticker cap.
    """
    clamped: dict[str, float] = {}
    clamps: list[ClampEvent] = []

    for ticker in sorted(weights):
        w = weights[ticker]
        cap = limits.max_position_pct
        if abs(w) > cap:
            new_w = cap if w > 0 else -cap
            clamps.append(ClampEvent(
                limit="max_position_pct", ticker=ticker, before=w, after=new_w,
            ))
            clamped[ticker] = new_w
        else:
            clamped[ticker] = w

    gross = sum(abs(w) for w in clamped.values())
    if gross > limits.max_gross_exposure:
        scale = limits.max_gross_exposure / gross
        clamped = {t: w * scale for t, w in clamped.items()}
        clamps.append(ClampEvent(
            limit="max_gross_exposure", before=gross, after=limits.max_gross_exposure,
        ))

    short_gross = sum(-w for w in clamped.values() if w < 0)
    if limits.max_short_exposure is not None and short_gross > limits.max_short_exposure:
        scale = limits.max_short_exposure / short_gross
        clamped = {t: (w * scale if w < 0 else w) for t, w in clamped.items()}
        clamps.append(ClampEvent(
            limit="max_short_exposure", before=short_gross,
            after=limits.max_short_exposure,
        ))

    net = sum(clamped.values())
    if limits.max_net_exposure is not None and abs(net) > limits.max_net_exposure:
        cap = limits.max_net_exposure
        if net > 0:
            positive = sum(w for w in clamped.values() if w > 0)
            negative = sum(w for w in clamped.values() if w < 0)
            target_positive = cap - negative
            scale = target_positive / positive if positive else 0.0
            clamped = {t: (w * scale if w > 0 else w) for t, w in clamped.items()}
        else:
            positive = sum(w for w in clamped.values() if w > 0)
            negative_abs = sum(-w for w in clamped.values() if w < 0)
            target_negative_abs = cap + positive
            scale = target_negative_abs / negative_abs if negative_abs else 0.0
            clamped = {t: (w * scale if w < 0 else w) for t, w in clamped.items()}
        clamps.append(ClampEvent(
            limit="max_net_exposure", before=net, after=sum(clamped.values()),
        ))

    return RiskResult(weights=clamped, clamps=clamps)
