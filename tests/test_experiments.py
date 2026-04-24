import pandas as pd

import ai_leader.experiments as experiments
from ai_leader.clients import FakeClient
from ai_leader.experiments import (
    ModelRun,
    build_comparison_table,
    build_model_comparison_dataframe,
    evaluate_model_on_dataframe,
    run_model_comparison,
    select_best_model,
)


def test_run_model_comparison_with_fake():
    df = pd.DataFrame(
        {
            "row_id": [0, 1],
            "Request Text": ["Test 1", "Test 2"],
            "Submission Channel": ["Email", "Chat"],
            "Order History": [None, None],
            "Category": ["Payment", "Delivery"],
            "Routing to Department": ["Customer Support", "Logistics"],
            "[Human] Initial Response": ["Thanks", "Thanks"],
        }
    )
    results = run_model_comparison(
        df=df,
        models=["fake-model"],
        client=FakeClient(),
        use_progress=False,
    )
    table = build_comparison_table(results)
    assert not table.empty


def _minimal_run(
    model: str,
    *,
    misroute: float,
    dept_acc: float,
    cost: float | None,
    median_ms: float | None,
) -> ModelRun:
    return ModelRun(
        model=model,
        predictions=pd.DataFrame(),
        usage={"prompt_tokens": 0, "completion_tokens": 0},
        latency_ms=[],
        quality_metrics={
            "row_count": 1,
            "misroute_rate": misroute,
            "department_accuracy": dept_acc,
            "category_accuracy": 0.0,
            "exact_route_match_rate": 0.0,
        },
        confidence_metrics={},
        cost_metrics={"cost_per_message_usd": cost, "cost_source": "unknown"},
        latency_metrics={"median_latency_ms": median_ms, "p95_latency_ms": median_ms},
    )


def test_select_best_model_prefers_lower_misroute() -> None:
    runs = {
        "high": _minimal_run("high", misroute=0.2, dept_acc=0.9, cost=0.01, median_ms=100.0),
        "low": _minimal_run("low", misroute=0.1, dept_acc=0.8, cost=0.01, median_ms=100.0),
    }
    assert select_best_model(runs) == "low"


def test_build_model_comparison_dataframe_includes_safety_columns() -> None:
    df = pd.DataFrame(
        {
            "row_id": [0, 1],
            "Request Text": ["Test 1", "Test 2"],
            "Submission Channel": ["Email", "Chat"],
            "Order History": [None, None],
            "Category": ["Payment", "Delivery"],
            "Routing to Department": ["Customer Support", "Logistics"],
            "[Human] Initial Response": ["Thanks", "Thanks"],
        }
    )
    results = run_model_comparison(
        df=df,
        models=["fake-model"],
        client=FakeClient(),
        use_progress=False,
    )
    table = build_model_comparison_dataframe(df, results)
    assert "unsafe_auto_route_rate" in table.columns
    assert "auto_route_coverage" in table.columns
    assert len(table) == 1


def test_select_best_model_tiebreak_department_accuracy() -> None:
    runs = {
        "b": _minimal_run("b", misroute=0.1, dept_acc=0.85, cost=0.01, median_ms=100.0),
        "a": _minimal_run("a", misroute=0.1, dept_acc=0.9, cost=0.01, median_ms=100.0),
    }
    assert select_best_model(runs) == "a"


def test_evaluate_model_on_dataframe_full_rows():
    df = pd.DataFrame(
        {
            "row_id": [0, 1],
            "Request Text": ["Test 1", "Test 2"],
            "Submission Channel": ["Email", "Chat"],
            "Order History": [None, None],
            "Category": ["Payment", "Delivery"],
            "Routing to Department": ["Customer Support", "Logistics"],
            "[Human] Initial Response": ["Thanks", "Thanks"],
        }
    )
    run = evaluate_model_on_dataframe(
        df=df,
        model="fake-model",
        client=FakeClient(),
        use_progress=False,
    )
    assert int(run.quality_metrics["row_count"]) == len(df)


def test_zero_prediction_run_keeps_estimated_cost_for_known_model() -> None:
    df = pd.DataFrame(
        [
            {
                "row_id": 0,
                "Request Text": "Need help with order CK-1",
                "Submission Channel": "Email",
                "Routing to Department": "Customer Support",
                "Category": "General Feedback",
            },
        ]
    )

    run = experiments._model_run_from_extraction(
        model="deepseek-ai/DeepSeek-V3.2",
        df=df,
        predictions=pd.DataFrame(),
        usage={"prompt_tokens": 0, "completion_tokens": 0},
        latency_ms=[],
        monthly_messages=20_000,
    )

    assert run.quality_metrics["row_count"] == 0
    assert float(run.cost_metrics["cost_per_message_usd"]) > 0
    assert float(run.cost_metrics["monthly_cost_usd"]) > 0
    assert run.cost_metrics["cost_source"] == "estimated"
