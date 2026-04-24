import pandas as pd

from ai_leader.evaluation import (
    compute_confidence_policy_metrics,
    compute_cost_projection,
    compute_latency_summary,
    compute_quality_metrics,
)


def test_compute_quality_metrics_department_only_misroute():
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
            # Category can be wrong while department routing is still correct.
            "category": ["Payment", "Returns"],
            "[Agent] Routing to Department": ["Customer Support", "Logistics"],
        }
    )
    metrics = compute_quality_metrics(gt_df, pred_df)
    assert metrics["exact_route_match_rate"] == 0.5
    # H2 misroute_rate is department-only (ignore category correctness).
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
            # For high-confidence auto-routing, we only care about department
            # correctness (category mismatch should not count as an error).
            "category": ["Returns", "Delivery"],
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
        correct_route_rate=0.5,
    )
    assert projection["monthly_cost_usd"] > 0
    assert projection["avg_prompt_tokens"] > 0
    assert projection["avg_completion_tokens"] > 0
    assert projection["cost_per_correct_route_usd"] is not None


def test_compute_latency_summary():
    summary = compute_latency_summary([100, 200, 300])
    assert summary["median_latency_ms"] == 200
    assert summary["p95_latency_ms"] >= 200
    assert summary["mean_latency_ms"] == 200
