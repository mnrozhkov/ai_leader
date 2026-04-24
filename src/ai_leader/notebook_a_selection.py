"""Notebook A (spec §5.2): Phase 1 model ranking and decision_summary.json shape."""

from __future__ import annotations

import math
from typing import Any


def _latency_sort_key(raw: object) -> float:
    """Missing / unknown latency sorts last (worst) for tie-breaking."""
    if raw is None:
        return float("inf")
    if raw is not None and isinstance(raw, float) and math.isnan(raw):
        return float("inf")
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("inf")


def select_notebook_a_phase1(comparison_rows: list[dict[str, Any]]) -> tuple[str, str]:
    """Pick the best model per docs/module_3_3_notebook_a.md §5.2.

    Order: highest department_accuracy → lower misroute_rate → lower
    cost_per_message_usd → lower median_latency_ms (unknown last).

    Final tie-break: lexicographic ``model`` id for stability.
    """
    if not comparison_rows:
        msg = "No comparison rows; run at least one model."
        raise ValueError(msg)
    rows = list(comparison_rows)
    if len(rows) == 1:
        m = str(rows[0]["model"])
        return m, "only candidate in this run"

    def sort_key(r: dict[str, Any]) -> tuple:
        return (
            -float(r["department_accuracy"]),
            float(r["misroute_rate"]),
            float(r["cost_per_message_usd"]),
            _latency_sort_key(r.get("median_latency_ms")),
            str(r["model"]),
        )

    ranked = sorted(rows, key=sort_key)
    best = ranked[0]
    model = str(best["model"])
    reason = (
        "highest department accuracy among candidates; "
        "tie-breakers: lower misroute rate, lower cost per message, "
        "lower median latency (unknown latency last)"
    )
    return model, reason


def build_decision_summary(
    *,
    summary: dict[str, Any],
    short_rationale: str = "",
) -> dict[str, Any]:
    """Build decision_summary.json payload (spec §6)."""
    decision = summary["decision"]
    dims = {str(d["name"]): d for d in decision["dimensions"]}

    def verdict(*names: str) -> dict[str, Any]:
        dim = next((dims[n] for n in names if n in dims), None)
        if dim is None:
            raise KeyError(f"Missing expected decision dimension; tried {names}")
        d = dim
        return {
            "status": d["status"],
            "value": d["value"],
            "threshold_pass": d["threshold_pass"],
        }

    return {
        "model": summary["model"],
        "row_count": summary["row_count"],
        "routing_verdict": verdict("department_accuracy", "routing_quality"),
        "safety_verdict": verdict("unsafe_auto_route_rate", "safety"),
        "cost_verdict": verdict("monthly_cost_usd", "cost"),
        "speed_verdict": verdict("p95_latency_ms", "speed"),
        "final_recommendation": decision["recommendation"],
        "short_rationale": short_rationale,
    }
