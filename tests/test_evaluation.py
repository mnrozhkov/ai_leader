import pandas as pd

from ai_leader.evaluation import (
    compute_confidence_policy_metrics,
    compute_cost_projection,
    compute_latency_summary,
    compute_quality_metrics,
)


def test_compute_quality_metrics_exact_match():
    gt_df = pd.DataFrame(
        {
            "row_id": [0, 1],
            "Category": ["Payment", "Delivery"],
            "Routing to Department": ["Customer Support", "Logistics"],
        }
    )
    pred_df = pd.DataFrame(
        {
            "row_id": [0, 1],
            "category": ["Payment", "Delivery"],
            "[Agent] Routing to Department": ["Customer Support", "Logistics"],
        }
    )
    metrics = compute_quality_metrics(gt_df, pred_df)
    assert metrics["exact_route_match_rate"] == 1.0
    assert metrics["misroute_rate"] == 0.0


def test_compute_confidence_policy_metrics():
    gt_df = pd.DataFrame(
        {
            "row_id": [0, 1],
            "Category": ["Payment", "Delivery"],
            "Routing to Department": ["Customer Support", "Logistics"],
        }
    )
    pred_df = pd.DataFrame(
        {
            "row_id": [0, 1],
            "category": ["Payment", "Delivery"],
            "[Agent] Routing to Department": ["Customer Support", "Returns"],
            "Confidence": ["High", "Low"],
        }
    )
    metrics = compute_confidence_policy_metrics(gt_df, pred_df)
    assert metrics["high_confidence_coverage"] == 0.5
    assert metrics["high_confidence_error_rate"] == 0.0


def test_compute_cost_projection():
    projection = compute_cost_projection(
        usage={"prompt_tokens": 1000, "completion_tokens": 2000},
        model="openai/gpt-oss-120b",
        row_count=10,
        monthly_messages=100,
        exact_match_rate=0.5,
    )
    assert projection["monthly_cost_usd"] > 0
    assert projection["cost_per_exact_match_usd"] is not None


def test_compute_latency_summary():
    summary = compute_latency_summary([100, 200, 300])
    assert summary["median_latency_ms"] == 200
    assert summary["p95_latency_ms"] >= 200
