import pandas as pd

from ai_leader.clients import FakeClient
from ai_leader.experiments import (
    build_comparison_table,
    evaluate_model_on_dataframe,
    run_model_comparison,
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
