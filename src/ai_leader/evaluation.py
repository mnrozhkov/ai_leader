"""Metric calculations for routing quality, cost, and latency."""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .config import H2_MISROUTE_MAX, MODEL_REGISTRY


def compute_exact_match_rate(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    joined = pd.merge(gt_df, pred_df, on="row_id", how="inner", suffixes=("", "_pred"))
    if joined.empty:
        return 0.0
    category_match = joined["Category"] == joined["category"]
    dept_match = joined["Routing to Department"] == joined["[Agent] Routing to Department"]
    return float((category_match & dept_match).mean())


def compute_misroute_rate(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    return 1.0 - compute_exact_match_rate(gt_df, pred_df)


def compute_accuracy_f1(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compute_quality_metrics(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict[str, object]:
    joined = pd.merge(gt_df, pred_df, on="row_id", how="inner", suffixes=("", "_pred"))
    if joined.empty:
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
        "misroute_rate": 1.0 - exact_match,
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
    joined = pd.merge(gt_df, pred_df, on="row_id", how="inner", suffixes=("", "_pred"))
    if joined.empty or confidence_col not in joined.columns:
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
    correct = (high_df["Category"] == high_df["category"]) & (
        high_df["Routing to Department"] == high_df["[Agent] Routing to Department"]
    )
    error_rate = 1.0 - float(correct.mean())
    return {
        "high_confidence_coverage": coverage,
        "high_confidence_error_rate": error_rate,
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


def compute_cost_projection(
    *,
    usage: dict[str, int],
    model: str,
    row_count: int,
    monthly_messages: int,
    exact_match_rate: float | None,
) -> dict[str, float]:
    total_cost = compute_cost(usage=usage, model=model)
    cost_per_message = total_cost / row_count if row_count else 0.0
    monthly_cost = cost_per_message * monthly_messages
    annual_cost = monthly_cost * 12
    cost_per_exact = None
    if exact_match_rate is not None and exact_match_rate > 0:
        cost_per_exact = cost_per_message / exact_match_rate
    return {
        "cost_per_message_usd": cost_per_message,
        "monthly_cost_usd": monthly_cost,
        "annual_cost_usd": annual_cost,
        "cost_per_exact_match_usd": cost_per_exact,
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


def compute_latency_summary(latency_ms: Iterable[float]) -> dict[str, float]:
    values = [v for v in latency_ms if v is not None]
    if not values:
        return {"median_latency_ms": 0.0, "p95_latency_ms": 0.0}
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 0:
        median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    else:
        median = sorted_vals[mid]
    return {"median_latency_ms": float(median), "p95_latency_ms": _percentile(sorted_vals, 0.95)}


def is_h2_pass(misroute_rate: float) -> bool:
    return misroute_rate <= H2_MISROUTE_MAX
