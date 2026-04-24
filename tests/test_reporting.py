"""Tests for reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest

from ai_leader.decision import Thresholds, evaluate_decision
from ai_leader.reporting import (
    display_evaluation_results,
    mvp_decision_summary_table,
    normalize_confusion_rows,
    quality_metrics_summary_table,
    show_figure,
)


def test_normalize_confusion_rows() -> None:
    assert normalize_confusion_rows([[2, 0], [0, 3]]) == [[1.0, 0.0], [0.0, 1.0]]
    assert normalize_confusion_rows([[0, 0]]) == [[0.0, 0.0]]


def test_quality_metrics_summary_table() -> None:
    df = quality_metrics_summary_table(
        {
            "misroute_rate": 0.1,
            "department_accuracy": 0.9,
            "category_accuracy": 0.85,
            "row_count": 42,
        },
    )
    assert df.shape == (1, 4)
    assert float(df["Misroute rate"].iloc[0]) == 0.1


def test_mvp_decision_summary_table() -> None:
    decision = evaluate_decision(
        quality_metrics={"department_accuracy": 0.82, "category_accuracy": 0.8},
        safety_metrics={"unsafe_auto_route_rate": 0.02},
        cost_metrics={"monthly_cost_usd": 900.0},
        latency_metrics={"p95_latency_ms": 100.0},
        thresholds=Thresholds(),
    )
    tab = mvp_decision_summary_table(decision)
    assert len(tab) == 5
    assert "Status" in tab.columns


@patch("IPython.display.display", autospec=True)
def test_show_figure_displays_and_closes(mock_display: object) -> None:
    fig, _ = plt.subplots()
    show_figure(fig)
    mock_display.assert_called_once()


def test_display_evaluation_results_prints_metrics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    @dataclass
    class _Stub:
        quality_metrics: dict[str, object]

    stub = _Stub(
        quality_metrics={
            "row_count": 2,
            "exact_route_match_rate": 0.5,
            "misroute_rate": 0.5,
            "category_accuracy": 1.0,
            "category_f1_macro": 1.0,
            "department_accuracy": 0.5,
            "department_f1_macro": 0.5,
            "confusion_category": None,
            "confusion_department": None,
        }
    )
    display_evaluation_results(eval_results=stub)
    out = capsys.readouterr().out
    assert "category_accuracy" in out
    assert "2" in out
