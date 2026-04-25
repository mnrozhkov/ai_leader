"""MVP decision logic for routing evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DecisionStatus = Literal["pass", "fail", "unknown"]


@dataclass(frozen=True)
class Thresholds:
    """Pass thresholds for default hypothesis-validation metrics."""

    department_accuracy_pass: float = 0.85
    category_accuracy_pass: float = 0.85
    unsafe_auto_route_rate_pass: float = 0.03
    monthly_cost_usd_pass: float = 1_000.0
    p95_latency_ms_pass: float = 5_000.0


def _classify(value: float, pass_max: float) -> DecisionStatus:
    if value <= pass_max:
        return "pass"
    return "fail"


def _classify_min(value: float, pass_min: float) -> DecisionStatus:
    if value >= pass_min:
        return "pass"
    return "fail"


def _to_float(value: object, *, default: float) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return default
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class DimensionResult:
    name: str
    status: DecisionStatus
    value: float
    threshold_pass: float


@dataclass(frozen=True)
class DecisionSummary:
    dimensions: list[DimensionResult] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "dimensions": [
                {
                    "name": d.name,
                    "status": d.status,
                    "value": d.value,
                    "threshold_pass": d.threshold_pass,
                }
                for d in self.dimensions
            ],
            "recommendation": self.recommendation,
        }


def evaluate_decision(
    run: object | None = None,
    *,
    thresholds: Thresholds | None = None,
    quality_metrics: dict[str, object] | None = None,
    safety_metrics: dict[str, float] | None = None,
    cost_metrics: dict[str, float | None | str] | None = None,
    latency_metrics: dict[str, float | None | str] | None = None,
) -> DecisionSummary:
    """Evaluate default notebook metrics and produce a recommendation.

    Dimensions (rows): department/category accuracy, unsafe auto-route rate,
    monthly cost, and p95 latency.
    """
    t = thresholds or Thresholds()

    def _safe_get_dict_metric(
        source: object,
        field_name: str,
        fallback: dict[str, object] | dict[str, float] | dict[str, float | None | str] | None,
    ) -> dict[str, object] | dict[str, float] | dict[str, float | None | str] | None:
        if isinstance(source, dict):
            value = source.get(field_name, fallback)
            return value if isinstance(value, dict) else fallback
        try:
            value = getattr(source, field_name)
        except Exception:
            return fallback
        return value if isinstance(value, dict) else fallback

    if run is not None:
        quality_metrics = _safe_get_dict_metric(run, "quality_metrics", quality_metrics)
        safety_metrics = _safe_get_dict_metric(run, "safety_metrics", safety_metrics)
        cost_metrics = _safe_get_dict_metric(run, "cost_metrics", cost_metrics)
        latency_metrics = _safe_get_dict_metric(run, "latency_metrics", latency_metrics)

    missing_metrics = quality_metrics is None or cost_metrics is None or latency_metrics is None
    if missing_metrics:
        raise ValueError(
            "evaluate_decision requires either a run object "
            "(with quality/safety/cost/latency metrics) or all metric dictionaries."
        )

    dept_acc = _to_float(quality_metrics.get("department_accuracy"), default=0.0)
    cat_acc = _to_float(quality_metrics.get("category_accuracy"), default=0.0)
    safety_known = safety_metrics is not None
    unsafe = float(safety_metrics.get("unsafe_auto_route_rate", 0.0)) if safety_known else 0.0
    raw_monthly_cost = cost_metrics.get("monthly_cost_usd")
    monthly_cost_known = raw_monthly_cost is not None
    monthly_cost = float(raw_monthly_cost) if monthly_cost_known else 0.0

    raw_p95_lat = latency_metrics.get("p95_latency_ms")
    p95_known = raw_p95_lat is not None
    p95_lat = float(raw_p95_lat) if p95_known else 0.0

    dims: list[DimensionResult] = [
        DimensionResult(
            name="department_accuracy",
            status=_classify_min(
                dept_acc,
                t.department_accuracy_pass,
            ),
            value=dept_acc,
            threshold_pass=t.department_accuracy_pass,
        ),
        DimensionResult(
            name="category_accuracy",
            status=_classify_min(
                cat_acc,
                t.category_accuracy_pass,
            ),
            value=cat_acc,
            threshold_pass=t.category_accuracy_pass,
        ),
        DimensionResult(
            name="unsafe_auto_route_rate",
            status=(
                _classify(unsafe, t.unsafe_auto_route_rate_pass) if safety_known else "unknown"
            ),
            value=unsafe,
            threshold_pass=t.unsafe_auto_route_rate_pass,
        ),
        DimensionResult(
            name="monthly_cost_usd",
            status=(
                _classify(monthly_cost, t.monthly_cost_usd_pass)
                if monthly_cost_known
                else "unknown"
            ),
            value=monthly_cost,
            threshold_pass=t.monthly_cost_usd_pass,
        ),
        DimensionResult(
            name="p95_latency_ms",
            status=(_classify(p95_lat, t.p95_latency_ms_pass) if p95_known else "unknown"),
            value=p95_lat,
            threshold_pass=t.p95_latency_ms_pass,
        ),
    ]

    statuses = [d.status for d in dims]
    known_statuses = [s for s in statuses if s != "unknown"]
    unknown_names = [d.name for d in dims if d.status == "unknown"]

    if all(s == "pass" for s in known_statuses) and not unknown_names:
        rec = "Proceed to MVP workflow"
    elif "fail" in known_statuses:
        fail_names = [d.name for d in dims if d.status == "fail"]
        suffix = f" (unknown: {', '.join(unknown_names)})" if unknown_names else ""
        rec = f"Improve and re-test (failing: {', '.join(fail_names)}){suffix}"
    elif unknown_names:
        rec = f"Proceed with guardrails (unknown: {', '.join(unknown_names)})"
    else:
        rec = "Proceed with guardrails"

    return DecisionSummary(dimensions=dims, recommendation=rec)
