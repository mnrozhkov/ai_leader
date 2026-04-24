"""Metric calculations for routing quality, cost, and latency."""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .config import H2_MISROUTE_MAX, MODEL_REGISTRY


def _safe_merge(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """Merge on row_id, returning None when pred_df lacks the column."""
    if pred_df.empty or "row_id" not in pred_df.columns:
        return None
    return pd.merge(
        gt_df,
        pred_df,
        on="row_id",
        how="inner",
        suffixes=("", "_pred"),
    )


def compute_exact_match_rate(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    joined = _safe_merge(gt_df, pred_df)
    if joined is None or joined.empty:
        return 0.0
    category_match = joined["Category"] == joined["category"]
    dept_match = joined["Routing to Department"] == joined["[Agent] Routing to Department"]
    return float((category_match & dept_match).mean())


def compute_misroute_rate(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    """
    Department-only misroute rate.

    This intentionally ignores category correctness and measures whether the predicted
    department equals the gold department.
    """
    joined = _safe_merge(gt_df, pred_df)
    if joined is None or joined.empty:
        return 0.0
    dept_match = joined["Routing to Department"] == joined["[Agent] Routing to Department"]
    dept_accuracy = float(dept_match.mean())
    return 1.0 - dept_accuracy


def compute_accuracy_f1(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compute_quality_metrics(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict[str, object]:
    joined = _safe_merge(gt_df, pred_df)
    if joined is None or joined.empty:
        return {
            "row_count": 0,
            "exact_route_match_rate": 0.0,
            "misroute_rate": 0.0,
            "category_accuracy": 0.0,
            "category_f1_macro": 0.0,
            "department_accuracy": 0.0,
            "department_f1_macro": 0.0,
            "confusion_category": None,
            "confusion_department": None,
        }

    y_cat_true = joined["Category"].tolist()
    y_cat_pred = joined["category"].tolist()
    y_dept_true = joined["Routing to Department"].tolist()
    y_dept_pred = joined["[Agent] Routing to Department"].tolist()

    cat_metrics = compute_accuracy_f1(y_cat_true, y_cat_pred)
    dept_metrics = compute_accuracy_f1(y_dept_true, y_dept_pred)
    exact_match = compute_exact_match_rate(gt_df, pred_df)

    cat_labels = sorted(set(y_cat_true) | set(y_cat_pred))
    dept_labels = sorted(set(y_dept_true) | set(y_dept_pred))
    return {
        "row_count": int(len(joined)),
        "exact_route_match_rate": exact_match,
        # H2 misroute rate is department-only (ignore category correctness).
        "misroute_rate": 1.0 - dept_metrics["accuracy"],
        "category_accuracy": cat_metrics["accuracy"],
        "category_f1_macro": cat_metrics["f1_macro"],
        "department_accuracy": dept_metrics["accuracy"],
        "department_f1_macro": dept_metrics["f1_macro"],
        "confusion_category": {
            "labels": cat_labels,
            "matrix": confusion_matrix(y_cat_true, y_cat_pred, labels=cat_labels).tolist(),
        },
        "confusion_department": {
            "labels": dept_labels,
            "matrix": confusion_matrix(y_dept_true, y_dept_pred, labels=dept_labels).tolist(),
        },
    }


def compute_confidence_policy_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    confidence_col: str = "Confidence",
    high_label: str = "High",
) -> dict[str, float]:
    joined = _safe_merge(gt_df, pred_df)
    if joined is None or joined.empty or confidence_col not in joined.columns:
        return {
            "high_confidence_coverage": 0.0,
            "high_confidence_error_rate": 0.0,
            "manual_review_rate": 1.0,
        }
    high_mask = joined[confidence_col].fillna("") == high_label
    total = len(joined)
    high_count = int(high_mask.sum())
    coverage = high_count / total if total else 0.0

    if high_count == 0:
        return {
            "high_confidence_coverage": coverage,
            "high_confidence_error_rate": 0.0,
            "manual_review_rate": 1.0 - coverage,
        }

    high_df = joined.loc[high_mask]
    # H2 is department-only: correctness of department routing defines whether a
    # high-confidence prediction can be auto-routed.
    correct = high_df["Routing to Department"] == high_df["[Agent] Routing to Department"]
    error_rate = 1.0 - float(correct.mean())
    return {
        "high_confidence_coverage": coverage,
        "high_confidence_error_rate": error_rate,
        "manual_review_rate": 1.0 - coverage,
    }


def compute_safety_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    confidence_col: str = "Confidence",
    high_label: str = "High",
) -> dict[str, float]:
    """Spec-aligned safety metrics for the confidence-based review policy.

    Returns keys that match the Notebook A spec table:
    - auto_route_coverage
    - auto_route_precision
    - unsafe_auto_route_rate
    - manual_review_rate
    """
    joined = _safe_merge(gt_df, pred_df)
    if joined is None:
        return {
            "auto_route_coverage": 0.0,
            "auto_route_precision": 0.0,
            "unsafe_auto_route_rate": 0.0,
            "manual_review_rate": 1.0,
        }
    total = len(joined)
    if total == 0 or confidence_col not in joined.columns:
        return {
            "auto_route_coverage": 0.0,
            "auto_route_precision": 0.0,
            "unsafe_auto_route_rate": 0.0,
            "manual_review_rate": 1.0,
        }

    high_mask = joined[confidence_col].fillna("") == high_label
    high_count = int(high_mask.sum())
    coverage = high_count / total

    if high_count == 0:
        return {
            "auto_route_coverage": 0.0,
            "auto_route_precision": 0.0,
            "unsafe_auto_route_rate": 0.0,
            "manual_review_rate": 1.0,
        }

    high_df = joined.loc[high_mask]
    correct = high_df["Routing to Department"] == high_df["[Agent] Routing to Department"]
    correct_count = int(correct.sum())
    wrong_count = high_count - correct_count

    return {
        "auto_route_coverage": coverage,
        "auto_route_precision": correct_count / high_count,
        "unsafe_auto_route_rate": wrong_count / total,
        "manual_review_rate": 1.0 - coverage,
    }


def compute_cost(
    *,
    usage: dict[str, int],
    model: str,
) -> float:
    pricing = MODEL_REGISTRY[model]
    return (
        usage["prompt_tokens"] * pricing["input"] / 1_000_000
        + usage["completion_tokens"] * pricing["output"] / 1_000_000
    )


_TYPICAL_PROMPT_TOKENS = 350
_TYPICAL_COMPLETION_TOKENS = 120


def estimate_cost_per_message(model: str) -> float:
    """Estimate per-message cost from registry pricing and typical token counts.

    Useful when real usage data is unavailable (e.g. skip-extraction mode).
    """
    entry = MODEL_REGISTRY.get(model)
    if entry is None:
        return 0.0
    input_price = float(entry["input"])
    output_price = float(entry["output"])
    return (
        _TYPICAL_PROMPT_TOKENS * input_price / 1_000_000
        + _TYPICAL_COMPLETION_TOKENS * output_price / 1_000_000
    )


def compute_cost_projection(
    *,
    usage: dict[str, int],
    model: str,
    row_count: int,
    monthly_messages: int,
    correct_route_rate: float | None,
) -> dict[str, float | None | str]:
    """Project per-message and monthly/annual costs.

    Fallback chain for ``cost_per_message_usd``:
    1. **measured** — computed from real token usage returned by the API.
    2. **estimated** — derived from typical token counts and registry pricing
       when real usage is unavailable (e.g. skip-extraction without metadata).

    The chosen source is recorded in ``cost_source``.
    """
    total_cost = compute_cost(usage=usage, model=model)
    cost_per_message = total_cost / row_count if row_count else 0.0

    has_real_usage = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0) > 0

    if has_real_usage and cost_per_message > 0:
        cost_source = "measured"
    else:
        cost_per_message = estimate_cost_per_message(model)
        cost_source = "estimated"

    monthly_cost = cost_per_message * monthly_messages
    annual_cost = monthly_cost * 12

    avg_prompt_tokens = usage["prompt_tokens"] / row_count if row_count else 0.0
    avg_completion_tokens = usage["completion_tokens"] / row_count if row_count else 0.0

    cost_per_correct_route: float | None = None
    if correct_route_rate is not None and correct_route_rate > 0:
        cost_per_correct_route = cost_per_message / correct_route_rate
    return {
        "cost_per_message_usd": cost_per_message,
        "avg_prompt_tokens": float(avg_prompt_tokens),
        "avg_completion_tokens": float(avg_completion_tokens),
        "monthly_cost_usd": monthly_cost,
        "annual_cost_usd": annual_cost,
        "cost_per_correct_route_usd": cost_per_correct_route,
        "cost_source": cost_source,
    }


def _percentile(values: Iterable[float], p: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return float(vals[0])
    rank = (len(vals) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(vals[low])
    weight = rank - low
    return float(vals[low] * (1 - weight) + vals[high] * weight)


def compute_latency_summary(
    latency_ms: Iterable[float],
) -> dict[str, float | None | str]:
    """Summarise per-row latency timings.

    When *latency_ms* is empty (no timing data available) every numeric field
    is ``None`` and ``latency_source`` is ``"unknown"``.  This avoids silently
    reporting ``0.0 ms`` which would look like an impossibly fast model.
    """
    values = [v for v in latency_ms if v is not None]
    if not values:
        return {
            "median_latency_ms": None,
            "p95_latency_ms": None,
            "mean_latency_ms": None,
            "latency_source": "unknown",
        }
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 0:
        median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    else:
        median = sorted_vals[mid]
    mean = sum(sorted_vals) / len(sorted_vals)
    return {
        "median_latency_ms": float(median),
        "p95_latency_ms": _percentile(sorted_vals, 0.95),
        "mean_latency_ms": float(mean),
        "latency_source": "measured",
    }


def compute_estimated_avg_routing_time_seconds(
    *,
    median_latency_ms: float,
    auto_route_rate: float,
    manual_review_time_assumption_seconds: float,
) -> float:
    """
    Business proxy for first routing/acknowledgement time.

    estimated_avg_routing_time_seconds =
        (auto_route_rate × median_model_latency_seconds)
        + ((1 - auto_route_rate) × manual_review_time_assumption_seconds)
    """
    auto_rate = max(0.0, min(1.0, float(auto_route_rate)))
    median_latency_seconds = float(median_latency_ms) / 1000.0
    manual_review_time_seconds = float(manual_review_time_assumption_seconds)
    return auto_rate * median_latency_seconds + (1.0 - auto_rate) * manual_review_time_seconds


def is_h2_pass(misroute_rate: float) -> bool:
    return misroute_rate <= H2_MISROUTE_MAX
