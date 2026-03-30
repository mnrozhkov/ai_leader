"""Tests for reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ai_leader.reporting import display_evaluation_results, normalize_confusion_rows


def test_normalize_confusion_rows() -> None:
    assert normalize_confusion_rows([[2, 0], [0, 3]]) == [[1.0, 0.0], [0.0, 1.0]]
    assert normalize_confusion_rows([[0, 0]]) == [[0.0, 0.0]]


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
