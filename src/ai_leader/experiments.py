"""High-level orchestration used by the notebook."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
from tqdm.auto import tqdm

from .clients import FakeClient, create_token_factory_client, run_extraction_async
from .config import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MONTHLY_MESSAGES,
    DEFAULT_TEMPERATURE,
    MODEL_REGISTRY,
    get_base_url,
)
from .evaluation import (
    compute_confidence_policy_metrics,
    compute_cost_projection,
    compute_latency_summary,
    compute_quality_metrics,
    compute_safety_metrics,
)
from .prompts import DEFAULT_SYSTEM_PROMPT
from .reporting import round_numeric_frame

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelRun:
    model: str
    predictions: pd.DataFrame
    usage: dict[str, int]
    latency_ms: list[float]
    quality_metrics: dict[str, object]
    safety_metrics: dict[str, float]
    confidence_metrics: dict[str, float]
    cost_metrics: dict[str, float | None | str]
    latency_metrics: dict[str, float | None | str]


def _model_run_from_extraction(
    *,
    model: str,
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    usage: dict[str, int],
    latency_ms: list[float],
    monthly_messages: int,
) -> ModelRun:
    if predictions.empty or "row_id" not in predictions.columns:
        log.warning(
            "Model %s: no valid predictions — returning zero metrics",
            model,
        )
        if model in MODEL_REGISTRY:
            cost_metrics = compute_cost_projection(
                usage=usage,
                model=model,
                row_count=0,
                monthly_messages=monthly_messages,
                correct_route_rate=None,
            )
        else:
            cost_metrics = {
                "cost_per_message_usd": 0.0,
                "avg_prompt_tokens": 0.0,
                "avg_completion_tokens": 0.0,
                "monthly_cost_usd": 0.0,
                "annual_cost_usd": 0.0,
                "cost_per_correct_route_usd": None,
                "cost_source": "unknown",
            }
        zero_quality: dict[str, object] = {
            "row_count": 0,
            "exact_route_match_rate": 0.0,
            "misroute_rate": 1.0,
            "category_accuracy": 0.0,
            "category_f1_macro": 0.0,
            "department_accuracy": 0.0,
            "department_f1_macro": 0.0,
            "confusion_category": None,
            "confusion_department": None,
        }
        return ModelRun(
            model=model,
            predictions=predictions,
            usage=usage,
            latency_ms=latency_ms,
            quality_metrics=zero_quality,
            safety_metrics={
                "auto_route_coverage": 0.0,
                "auto_route_precision": 0.0,
                "unsafe_auto_route_rate": 0.0,
                "manual_review_rate": 1.0,
            },
            confidence_metrics={
                "high_confidence_coverage": 0.0,
                "high_confidence_error_rate": 0.0,
                "manual_review_rate": 1.0,
            },
            cost_metrics=cost_metrics,
            latency_metrics=compute_latency_summary(latency_ms),
        )

    quality_metrics = compute_quality_metrics(df, predictions)
    safety_metrics = compute_safety_metrics(df, predictions)
    confidence_metrics = compute_confidence_policy_metrics(df, predictions)
    if model in MODEL_REGISTRY:
        row_count = int(quality_metrics["row_count"])
        department_accuracy = float(quality_metrics["department_accuracy"])
        cost_metrics = compute_cost_projection(
            usage=usage,
            model=model,
            row_count=row_count,
            monthly_messages=monthly_messages,
            # H6 unit economics are anchored to department-level routing quality.
            correct_route_rate=department_accuracy,
        )
    else:
        row_count = int(quality_metrics["row_count"])
        avg_prompt_tokens = usage["prompt_tokens"] / row_count if row_count else 0.0
        avg_completion_tokens = usage["completion_tokens"] / row_count if row_count else 0.0
        cost_metrics = {
            "cost_per_message_usd": 0.0,
            "avg_prompt_tokens": float(avg_prompt_tokens),
            "avg_completion_tokens": float(avg_completion_tokens),
            "monthly_cost_usd": 0.0,
            "annual_cost_usd": 0.0,
            "cost_per_correct_route_usd": None,
            "cost_source": "unknown",
        }
    latency_metrics = compute_latency_summary(latency_ms)
    return ModelRun(
        model=model,
        predictions=predictions,
        usage=usage,
        latency_ms=latency_ms,
        quality_metrics=quality_metrics,
        safety_metrics=safety_metrics,
        confidence_metrics=confidence_metrics,
        cost_metrics=cost_metrics,
        latency_metrics=latency_metrics,
    )


def _make_progress_bar(desc: str, total: int) -> Any:
    """``tqdm.auto`` bar (notebook-aware) with ``.update(n)`` / ``.close()`` for extraction."""

    return tqdm(
        total=total,
        desc=desc,
        unit="row",
        dynamic_ncols=True,
        leave=True,
    )


async def evaluate_model_on_dataframe_async(
    *,
    df: pd.DataFrame,
    model: str,
    client,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    few_shot_df: pd.DataFrame | None = None,
    monthly_messages: int = DEFAULT_MONTHLY_MESSAGES,
    use_progress: bool = True,
) -> ModelRun:
    progress_bar = _make_progress_bar(f"Running {model}", len(df)) if use_progress else None
    try:
        predictions, usage, latency_ms = await run_extraction_async(
            df=df,
            few_shot_df=few_shot_df,
            max_concurrency=max_concurrency,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            progress_bar=progress_bar,
            client=client,
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return _model_run_from_extraction(
        model=model,
        df=df,
        predictions=predictions,
        usage=usage,
        latency_ms=latency_ms,
        monthly_messages=monthly_messages,
    )


def evaluate_model_on_dataframe(
    *,
    df: pd.DataFrame,
    model: str,
    client,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    few_shot_df: pd.DataFrame | None = None,
    monthly_messages: int = DEFAULT_MONTHLY_MESSAGES,
    use_progress: bool = True,
) -> ModelRun:
    coro = evaluate_model_on_dataframe_async(
        df=df,
        model=model,
        client=client,
        system_prompt=system_prompt,
        temperature=temperature,
        max_concurrency=max_concurrency,
        few_shot_df=few_shot_df,
        monthly_messages=monthly_messages,
        use_progress=use_progress,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "evaluate_model_on_dataframe() uses asyncio.run() and cannot run while an event loop "
        "is already active (e.g. in Jupyter after `await`). Use "
        "`await evaluate_model_on_dataframe_async(...)` with the same arguments instead."
    ) from None


async def run_model_comparison_async(
    *,
    df: pd.DataFrame,
    models: Sequence[str],
    api_key: str | None = None,
    client=None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    few_shot_df: pd.DataFrame | None = None,
    monthly_messages: int = DEFAULT_MONTHLY_MESSAGES,
    use_progress: bool = True,
) -> dict[str, ModelRun]:
    """Run evaluation across multiple models.

    When *api_key* is provided (and *client* is None), a fresh client is created per model (separate
    HTTP connection state); the base URL is still the same global Token Factory host unless you pass
    a custom *client* or set ``TOKENFACTORY_BASE_URL``. If a single *client* is passed it is reused
    for all models.
    """
    results: dict[str, ModelRun] = {}
    for model in models:
        if client is not None:
            model_client = client
        elif api_key:
            model_client = create_client(api_key, model=model)
        else:
            raise ValueError("Either api_key or client must be provided.")
        try:
            results[model] = await evaluate_model_on_dataframe_async(
                df=df,
                model=model,
                client=model_client,
                system_prompt=system_prompt,
                temperature=temperature,
                max_concurrency=max_concurrency,
                few_shot_df=few_shot_df,
                monthly_messages=monthly_messages,
                use_progress=use_progress,
            )
        except Exception as exc:
            log.error(
                "Model %s failed — skipping (%s: %s)",
                model,
                type(exc).__name__,
                exc,
            )
    return results


def run_model_comparison(
    *,
    df: pd.DataFrame,
    models: Sequence[str],
    api_key: str | None = None,
    client=None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    few_shot_df: pd.DataFrame | None = None,
    monthly_messages: int = DEFAULT_MONTHLY_MESSAGES,
    use_progress: bool = True,
) -> dict[str, ModelRun]:
    return asyncio.run(
        run_model_comparison_async(
            df=df,
            models=models,
            api_key=api_key,
            client=client,
            system_prompt=system_prompt,
            temperature=temperature,
            max_concurrency=max_concurrency,
            few_shot_df=few_shot_df,
            monthly_messages=monthly_messages,
            use_progress=use_progress,
        )
    )


def build_comparison_table(model_runs: dict[str, ModelRun]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, run in model_runs.items():
        rows.append(
            {
                "model": model,
                "exact_match_rate": run.quality_metrics["exact_route_match_rate"],
                "misroute_rate": run.quality_metrics["misroute_rate"],
                "category_accuracy": run.quality_metrics["category_accuracy"],
                "department_accuracy": run.quality_metrics["department_accuracy"],
                "cost_per_message_usd": run.cost_metrics["cost_per_message_usd"],
                "cost_per_correct_route_usd": run.cost_metrics["cost_per_correct_route_usd"],
                "monthly_cost_usd": run.cost_metrics["monthly_cost_usd"],
                "annual_cost_usd": run.cost_metrics["annual_cost_usd"],
                "avg_prompt_tokens": run.cost_metrics["avg_prompt_tokens"],
                "avg_completion_tokens": run.cost_metrics["avg_completion_tokens"],
                "median_latency_ms": run.latency_metrics["median_latency_ms"],
                "mean_latency_ms": run.latency_metrics["mean_latency_ms"],
                "p95_latency_ms": run.latency_metrics["p95_latency_ms"],
            }
        )
    return round_numeric_frame(pd.DataFrame(rows))


def build_model_comparison_dataframe(
    df: pd.DataFrame,
    model_runs: dict[str, ModelRun],
    *,
    show_all: bool = False,
) -> pd.DataFrame:
    """Build the same columns as ``comparison_table.csv`` from ``run_notebook_a.py``.

    Default output is a compact metric subset used in notebooks.
    Set ``show_all=True`` for the full routing/cost/latency/safety column set.
    """
    rows: list[dict[str, object]] = []
    for mdl, run in model_runs.items():
        qm = run.quality_metrics
        sm = compute_safety_metrics(df, run.predictions)
        cm = run.cost_metrics
        lm = run.latency_metrics
        rows.append(
            {
                "model": mdl,
                "department_accuracy": qm["department_accuracy"],
                "misroute_rate": qm["misroute_rate"],
                "category_accuracy": qm["category_accuracy"],
                "cost_per_message_usd": cm["cost_per_message_usd"],
                "monthly_cost_usd": cm["monthly_cost_usd"],
                "cost_source": cm.get("cost_source", "unknown"),
                "median_latency_ms": lm["median_latency_ms"],
                "p95_latency_ms": lm["p95_latency_ms"],
                "latency_source": lm.get("latency_source", "unknown"),
                "auto_route_coverage": sm["auto_route_coverage"],
                "auto_route_precision": sm["auto_route_precision"],
                "unsafe_auto_route_rate": sm["unsafe_auto_route_rate"],
                "manual_review_rate": sm["manual_review_rate"],
            }
        )
    out_df = round_numeric_frame(pd.DataFrame(rows))
    if show_all:
        return out_df

    compact_columns = [
        "model",
        "department_accuracy",
        "category_accuracy",
        "unsafe_auto_route_rate",
        "monthly_cost_usd",
        "p95_latency_ms",
    ]
    present_columns = [col for col in compact_columns if col in out_df.columns]
    return out_df[present_columns]


def select_best_model(model_runs: dict[str, ModelRun]) -> str | None:
    """Pick the best candidate run.

    Order (per Notebook A / Module 3.3 spec): lowest **misroute rate**;
    ties broken by **higher** department accuracy, then **lower** cost per
    message, then **lower** median latency. Missing cost/latency sort as worst.
    """
    if not model_runs:
        return None

    def sort_key(run: ModelRun) -> tuple[float, float, float, float]:
        qm = run.quality_metrics
        cm = run.cost_metrics
        lm = run.latency_metrics
        misroute = float(cast(float | int, qm["misroute_rate"]))
        dept_acc = float(cast(float | int, qm["department_accuracy"]))
        raw_cost = cm["cost_per_message_usd"]
        cost = float(raw_cost) if raw_cost is not None else float("inf")
        med = lm["median_latency_ms"]
        latency = float(med) if med is not None else float("inf")
        return (misroute, -dept_acc, cost, latency)

    best = min(model_runs.values(), key=sort_key)
    return best.model


def compute_confidence_policy_metrics_for_run(run: ModelRun) -> dict[str, float]:
    return run.confidence_metrics


def compute_cost_projection_for_run(run: ModelRun) -> dict[str, float]:
    return run.cost_metrics


def compute_latency_summary_for_run(run: ModelRun) -> dict[str, float]:
    return run.latency_metrics


def create_client(
    api_key: str | None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    mode: str | None = "TOKEN_FACTORY",
) -> Any:
    """Create an API client.

    When *base_url* is not provided, uses ``TOKENFACTORY_BASE_URL`` from the environment if set,
    otherwise ``get_base_url(model)`` (normally the global Token Factory host for all models).
    """
    if mode == "FAKE":
        return FakeClient()
    if not api_key:
        raise ValueError("API key is required for real model runs.")
    if base_url is None:
        base_url = os.getenv("TOKENFACTORY_BASE_URL") or (
            get_base_url(model) if model else get_base_url("")
        )
    return create_token_factory_client(api_key, base_url=base_url)
