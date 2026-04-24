"""Tests for Notebook A Phase 1 selection (spec §5.2)."""

from __future__ import annotations

import pytest

from ai_leader.notebook_a_selection import (
    build_decision_summary,
    select_notebook_a_phase1,
)


def _row(
    model: str,
    dept_acc: float,
    misroute: float,
    cost: float,
    lat: float | None,
) -> dict:
    return {
        "model": model,
        "department_accuracy": dept_acc,
        "misroute_rate": misroute,
        "cost_per_message_usd": cost,
        "median_latency_ms": lat,
        "monthly_cost_usd": 0.0,
        "cost_source": "measured",
        "latency_source": "measured",
        "auto_route_coverage": 0.5,
        "auto_route_precision": 0.5,
        "unsafe_auto_route_rate": 0.1,
        "manual_review_rate": 0.5,
        "category_accuracy": 0.5,
    }


def test_select_single_candidate() -> None:
    rows = [_row("a/b", 0.9, 0.1, 0.01, 1000.0)]
    m, reason = select_notebook_a_phase1(rows)
    assert m == "a/b"
    assert "only candidate" in reason.lower()


def test_select_highest_department_accuracy() -> None:
    rows = [
        _row("m1", 0.7, 0.3, 0.001, 2000.0),
        _row("m2", 0.8, 0.2, 0.001, 2000.0),
    ]
    m, _ = select_notebook_a_phase1(rows)
    assert m == "m2"


def test_tiebreak_misroute() -> None:
    rows = [
        _row("m1", 0.8, 0.25, 0.001, 1000.0),
        _row("m2", 0.8, 0.15, 0.001, 1000.0),
    ]
    m, _ = select_notebook_a_phase1(rows)
    assert m == "m2"


def test_tiebreak_cost() -> None:
    rows = [
        _row("m1", 0.8, 0.2, 0.01, 1000.0),
        _row("m2", 0.8, 0.2, 0.001, 1000.0),
    ]
    m, _ = select_notebook_a_phase1(rows)
    assert m == "m2"


def test_tiebreak_latency_unknown_worse() -> None:
    rows = [
        _row("m1", 0.8, 0.2, 0.001, None),
        _row("m2", 0.8, 0.2, 0.001, 500.0),
    ]
    m, _ = select_notebook_a_phase1(rows)
    assert m == "m2"


def test_empty_raises() -> None:
    with pytest.raises(ValueError, match="No comparison"):
        select_notebook_a_phase1([])


def test_build_decision_summary() -> None:
    summary = {
        "model": "x/y",
        "row_count": 10,
        "decision": {
            "dimensions": [
                {
                    "name": "routing_quality",
                    "status": "fail",
                    "value": 0.28,
                    "threshold_pass": 0.1,
                    "threshold_borderline": 0.15,
                },
                {
                    "name": "safety",
                    "status": "fail",
                    "value": 0.22,
                    "threshold_pass": 0.03,
                    "threshold_borderline": 0.06,
                },
                {
                    "name": "cost",
                    "status": "pass",
                    "value": 0.01,
                    "threshold_pass": 0.05,
                    "threshold_borderline": 0.1,
                },
                {
                    "name": "speed",
                    "status": "pass",
                    "value": 1800.0,
                    "threshold_pass": 5000.0,
                    "threshold_borderline": 10000.0,
                },
            ],
            "recommendation": "Improve and re-test",
        },
    }
    out = build_decision_summary(summary=summary, short_rationale="")
    assert out["model"] == "x/y"
    assert out["row_count"] == 10
    assert out["routing_verdict"]["status"] == "fail"
    assert out["routing_verdict"]["value"] == 0.28
    assert out["final_recommendation"] == "Improve and re-test"
    assert out["short_rationale"] == ""
