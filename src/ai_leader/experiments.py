"""High-level orchestration used by the notebook."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from tqdm import tqdm

from .clients import FakeClient, create_token_factory_client, run_extraction_async
from .config import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MONTHLY_MESSAGES,
    DEFAULT_TEMPERATURE,
    H2_MISROUTE_MAX,
    MODEL_REGISTRY,
)
from .evaluation import (
    compute_confidence_policy_metrics,
    compute_cost_projection,
    compute_latency_summary,
    compute_quality_metrics,
)
from .prompts import DEFAULT_SYSTEM_PROMPT


@dataclass(slots=True)
class ModelRun:
    model: str
    predictions: pd.DataFrame
    usage: dict[str, int]
    latency_ms: list[float]
    quality_metrics: dict[str, object]
    confidence_metrics: dict[str, float]
    cost_metrics: dict[str, float]
    latency_metrics: dict[str, float]


def _model_run_from_extraction(
    *,
    model: str,
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    usage: dict[str, int],
    latency_ms: list[float],
    monthly_messages: int,
) -> ModelRun:
    quality_metrics = compute_quality_metrics(df, predictions)
    confidence_metrics = compute_confidence_policy_metrics(df, predictions)
    if model in MODEL_REGISTRY:
        cost_metrics = compute_cost_projection(
            usage=usage,
            model=model,
            row_count=quality_metrics["row_count"],
            monthly_messages=monthly_messages,
            exact_match_rate=quality_metrics["exact_route_match_rate"],
        )
    else:
        cost_metrics = {
            "cost_per_message_usd": 0.0,
            "monthly_cost_usd": 0.0,
            "annual_cost_usd": 0.0,
            "cost_per_exact_match_usd": None,
        }
    latency_metrics = compute_latency_summary(latency_ms)
    return ModelRun(
        model=model,
        predictions=predictions,
        usage=usage,
        latency_ms=latency_ms,
        quality_metrics=quality_metrics,
        confidence_metrics=confidence_metrics,
        cost_metrics=cost_metrics,
        latency_metrics=latency_metrics,
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
    progress_bar = tqdm(total=len(df), desc=f"Running {model}") if use_progress else None
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
    return asyncio.run(
        evaluate_model_on_dataframe_async(
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
    )


async def run_model_comparison_async(
    *,
    df: pd.DataFrame,
    models: Sequence[str],
    client,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    few_shot_df: pd.DataFrame | None = None,
    monthly_messages: int = DEFAULT_MONTHLY_MESSAGES,
    use_progress: bool = True,
) -> dict[str, ModelRun]:
    results: dict[str, ModelRun] = {}
    for model in models:
        results[model] = await evaluate_model_on_dataframe_async(
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
    return results


def run_model_comparison(
    *,
    df: pd.DataFrame,
    models: Sequence[str],
    client,
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
                "monthly_cost_usd": run.cost_metrics["monthly_cost_usd"],
                "median_latency_ms": run.latency_metrics["median_latency_ms"],
                "p95_latency_ms": run.latency_metrics["p95_latency_ms"],
            }
        )
    return pd.DataFrame(rows)


def select_best_model(model_runs: dict[str, ModelRun]) -> str | None:
    if not model_runs:
        return None
    acceptable: list[ModelRun] = [
        run
        for run in model_runs.values()
        if run.quality_metrics["misroute_rate"] <= H2_MISROUTE_MAX
    ]
    candidates = acceptable if acceptable else list(model_runs.values())
    return min(candidates, key=lambda run: run.cost_metrics["cost_per_message_usd"]).model


def compute_confidence_policy_metrics_for_run(run: ModelRun) -> dict[str, float]:
    return run.confidence_metrics


def compute_cost_projection_for_run(run: ModelRun) -> dict[str, float]:
    return run.cost_metrics


def compute_latency_summary_for_run(run: ModelRun) -> dict[str, float]:
    return run.latency_metrics


def create_client(
    api_key: str | None,
    *,
    client: str | None = None,
    base_url: str | None = None,
    mode: str | None = "TOKEN_FACTORY",
) -> Any:
    if mode == "FAKE":
        return FakeClient()
    if not api_key:
        raise ValueError("API key is required for real model runs.")
    if base_url is None:
        base_url = os.getenv("TOKENFACTORY_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
    return create_token_factory_client(api_key, base_url=base_url)
