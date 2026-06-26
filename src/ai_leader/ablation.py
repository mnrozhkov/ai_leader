"""Few-shot ablation helpers.

This module keeps few-shot experimental business logic out of notebooks:
- prepare a deterministic ablation slice
- run the async evaluation for multiple `n_shot` values
- pick the best `n_shot` and provide an error-review breakdown
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .clients import run_extraction_async
from .config import MODEL_REGISTRY
from .evaluation import compute_cost, compute_latency_summary, compute_quality_metrics
from .experiments import _make_progress_bar


@dataclass(slots=True)
class FewShotAblationConfig:
    """Few-shot ablation settings independent of prompt/model execution."""

    best_model: str
    few_shot_counts: list[int]
    ablation_random_state: int
    few_shot_pool_n: int
    ablation_eval_n: int


def prepare_few_shot_ablation_slice(
    df: pd.DataFrame,
    *,
    config: FewShotAblationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Split the sampled ablation slice into (few-shot pool, fixed eval set)."""
    abl_n = min(config.few_shot_pool_n + config.ablation_eval_n, len(df))
    if abl_n < 2:
        raise ValueError("Few-shot ablation needs at least two labeled rows.")

    df_ablation = (
        df.sample(n=abl_n, random_state=config.ablation_random_state).reset_index(drop=True).copy()
    )

    # Reserve >=1 eval row; cap pool by available rows (small fixtures / CI).
    pool_cap = min(config.few_shot_pool_n, abl_n - 1)
    few_shot_pool_df = df_ablation.iloc[:pool_cap].copy()
    eval_df_fixed = df_ablation.iloc[pool_cap:].copy()

    if eval_df_fixed.empty:
        raise ValueError(
            "Few-shot ablation needs at least one eval row after the few-shot pool; "
            f"increase labeled data or lower FEW_SHOT_POOL_N (pool={len(few_shot_pool_df)})."
        )

    few_shot_counts = [n for n in config.few_shot_counts if n <= len(few_shot_pool_df)]
    if not few_shot_counts:
        raise ValueError("No feasible n_shot values; enlarge the few-shot pool.")

    print(
        "Few-shot ablation slice: "
        f"pool={len(few_shot_pool_df)} rows, fixed eval={len(eval_df_fixed)} rows, "
        f"model={config.best_model}"
    )

    return few_shot_pool_df, eval_df_fixed, few_shot_counts


async def run_few_shot_ablation_async(
    *,
    few_shot_pool_df: pd.DataFrame,
    eval_df_fixed: pd.DataFrame,
    few_shot_counts: list[int],
    best_model: str,
    client: Any,
    max_concurrency: int = 5,
    system_prompt: str = "",
    temperature: float = 0.1,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Run the ablation for multiple `n_shot` values on the same eval slice."""
    phase2_predictions: dict[int, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []

    for idx, n_shot in enumerate(few_shot_counts, 1):
        few_df = few_shot_pool_df.head(n_shot) if n_shot > 0 else None
        short_model = best_model.rsplit("/", 1)[-1]
        label = (
            f"[Few-shot {idx}/{len(few_shot_counts)}] "
            f"{short_model}  n_shot={n_shot}  eval={len(eval_df_fixed)}"
        )

        progress = _make_progress_bar(label, len(eval_df_fixed)) if show_progress else None
        pred_df, usage, latency_ms = await run_extraction_async(
            df=eval_df_fixed,
            few_shot_df=few_df,
            max_concurrency=max_concurrency,
            model=best_model,
            system_prompt=system_prompt,
            temperature=temperature,
            progress_bar=progress,
            client=client,
        )
        if progress is not None:
            progress.close()

        qm = compute_quality_metrics(eval_df_fixed, pred_df)
        lat = compute_latency_summary(latency_ms)

        cost_total_usd = (
            compute_cost(usage=usage, model=best_model) if best_model in MODEL_REGISTRY else 0.0
        )
        n_eval = len(eval_df_fixed)
        cost_per_message_usd = cost_total_usd / n_eval if n_eval else 0.0

        rows.append(
            {
                "n_shot": n_shot,
                "department_accuracy": qm["department_accuracy"],
                "misroute_rate": qm["misroute_rate"],
                "department_f1_macro": qm["department_f1_macro"],
                "category_accuracy": qm["category_accuracy"],
                "category_f1_macro": qm["category_f1_macro"],
                "cost_per_message_usd": cost_per_message_usd,
                "cost_total_usd": cost_total_usd,
                "median_latency_ms": lat["median_latency_ms"],
                "mean_latency_ms": lat["mean_latency_ms"],
                "p95_latency_ms": lat["p95_latency_ms"],
            }
        )
        phase2_predictions[n_shot] = pred_df

    df_few_shot_ablation = pd.DataFrame(rows)
    return df_few_shot_ablation, phase2_predictions


def pick_optimal_n_shot(df_few_shot_ablation: pd.DataFrame) -> int:
    """Pick the best `n_shot` by (department_accuracy, department_f1_macro)."""
    if df_few_shot_ablation.empty:
        raise ValueError("df_few_shot_ablation is empty; nothing to pick from.")
    required = {"n_shot", "department_accuracy", "department_f1_macro"}
    missing = required - set(df_few_shot_ablation.columns)
    if missing:
        raise KeyError(f"Missing required columns in df_few_shot_ablation: {sorted(missing)}")

    return int(
        df_few_shot_ablation.sort_values(
            ["department_accuracy", "department_f1_macro"],
            ascending=[False, False],
        ).iloc[0]["n_shot"]
    )


def few_shot_department_error_review(
    *,
    eval_df_fixed: pd.DataFrame,
    phase2_predictions: dict[int, pd.DataFrame],
    optimal_n_shot: int,
    ad_col: str = "[Agent] Routing to Department",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (corrected, regressed) dept-routing cases vs the 0-shot baseline."""
    if 0 not in phase2_predictions:
        raise KeyError("phase2_predictions must include n_shot=0 for error review.")
    if optimal_n_shot not in phase2_predictions:
        raise KeyError("phase2_predictions must include the optimal_n_shot prediction.")

    p0_df = phase2_predictions[0]
    p_star_df = phase2_predictions[optimal_n_shot]
    required = {"row_id", ad_col}
    if not required.issubset(p0_df.columns) or p0_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    if not required.issubset(p_star_df.columns) or p_star_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    base = eval_df_fixed[["row_id", "Request Text", "Category", "Routing to Department"]].rename(
        columns={"Routing to Department": "dept_gt"}
    )

    p0 = p0_df[["row_id", ad_col]].rename(columns={ad_col: "pred_0shot"})
    p_star = p_star_df[["row_id", ad_col]].rename(columns={ad_col: "pred_best_n"})

    flips = base.merge(p0, on="row_id", how="left").merge(p_star, on="row_id", how="left")
    flips["ok_0"] = flips["dept_gt"] == flips["pred_0shot"]
    flips["ok_star"] = flips["dept_gt"] == flips["pred_best_n"]

    corrected = flips[(~flips["ok_0"]) & flips["ok_star"]]
    regressed = flips[flips["ok_0"] & (~flips["ok_star"])]

    display_cols = ["Request Text", "dept_gt", "pred_0shot", "pred_best_n"]
    return corrected[display_cols], regressed[display_cols]
