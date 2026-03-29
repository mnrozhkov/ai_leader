"""Reusable utilities for the Phase 3 notebook lab."""

from __future__ import annotations

import asyncio
import json
import math
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

MODEL_REGISTRY = {
    "deepseek-ai/DeepSeek-V3.2": {"input": 0.30, "output": 0.45},
    "zai-org/GLM-5": {"input": 1.00, "output": 3.20},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
}

MODELS = list(MODEL_REGISTRY)

DEFAULT_COLORS = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
]
MODEL_PALETTE = {m: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i, m in enumerate(MODELS)}

Category = Literal["Payment", "Order Issue", "Delivery", "General Feedback"]
Department = Literal["Customer Support", "Returns", "Product Team", "Logistics"]

EVAL_COLUMNS = {
    "Category": "category",
    "Routing to Department": "[Agent] Routing to Department",
}
METRIC_COLS = ["accuracy", "f1_macro"]


class SupportTicketExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Category = Field(description="Best-fit category for the customer request.")
    routing_to_department: Department = Field(
        description="Department that should handle the request.",
        alias="[Agent] Routing to Department",
    )
    agent_initial_response: str = Field(
        description="A professional first response to the customer.",
        alias="[Agent] Initial Response",
    )


response_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "support_ticket_extraction",
        "schema": SupportTicketExtraction.model_json_schema(),
        "strict": True,
    },
}


def short_name(model: str) -> str:
    """Derive a display-friendly short name from a full model id."""
    return model.rsplit("/", 1)[-1]


def model_id(model: str | object) -> str:
    """Resolve model id from a string or row-like object (e.g. pandas Series)."""
    if isinstance(model, str):
        return model
    mid = getattr(model, "model", None)
    if isinstance(mid, str):
        return mid
    return str(model)


def compute_cost(usage: dict[str, int], model: str) -> float:
    """Return total cost in USD from a usage dict and model name."""
    p = MODEL_REGISTRY[model]
    return (
        usage["prompt_tokens"] * p["input"] / 1_000_000
        + usage["completion_tokens"] * p["output"] / 1_000_000
    )


def build_user_prompt(
    request_text: str,
    submission_channel: str,
    order_history: str | None = None,
    few_shot_examples: list[dict] | None = None,
) -> str:
    parts: list[str] = []
    if few_shot_examples:
        parts.append("### Examples ###")
        for i, ex in enumerate(few_shot_examples, 1):
            out = {
                "Category": ex["Category"],
                "Routing to Department": ex["Routing to Department"],
                "[Agent] Initial Response": ex.get(
                    "[Agent] Initial Response",
                    ex.get("[Human] Initial Response", ""),
                ),
            }
            parts.append(
                f"Example {i}\nInput:\n"
                f"  Request Text: {ex['Request Text']}\n"
                f"  Submission Channel: {ex['Submission Channel']}\n"
                f"  Order History: {ex.get('Order History')}\n"
                f"Output:\n{json.dumps(out, ensure_ascii=False)}"
            )
    parts.append("### New case — classify and respond ###")
    parts.append(f"Request Text: {request_text}")
    parts.append(f"Submission Channel: {submission_channel}")
    parts.append(f"Order History: {order_history if order_history is not None else 'null'}")
    return "\n\n".join(parts)


async def extract_row(
    row: dict,
    few_shot_examples: list[dict] | None,
    model: str,
    semaphore: asyncio.Semaphore | None,
    system_prompt: str,
    temperature: float,
    client,
    max_retries: int = 3,
) -> tuple[dict, dict[str, int]]:
    """Run inference for a single row with retries; returns dict with row_id preserved."""
    user_prompt = build_user_prompt(
        request_text=row["Request Text"],
        submission_channel=row["Submission Channel"],
        order_history=row.get("Order History"),
        few_shot_examples=few_shot_examples,
    )

    async def _call() -> tuple[dict, dict[str, int]]:
        completion = await client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_schema,
        )
        content = (completion.choices[0].message.content or "").strip()
        if not content:
            raise ValueError(
                f"Model returned empty content. "
                f"Finish reason: {completion.choices[0].finish_reason}"
            )
        # Some models wrap JSON in markdown fences -> strip them
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        parsed = json.loads(content)
        validated = SupportTicketExtraction.model_validate(parsed)
        result = validated.model_dump(by_alias=True, mode="json")
        result["row_id"] = row["row_id"]

        usage = {
            "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(completion.usage, "completion_tokens", 0) or 0,
        }
        return result, usage

    async def _call_with_retries() -> tuple[dict, dict[str, int]]:
        for attempt in range(1, max_retries + 1):
            try:
                return await _call()
            except (ValueError, json.JSONDecodeError):
                if attempt == max_retries:
                    raise
                await asyncio.sleep(2**attempt)
        raise RuntimeError("Unreachable retry state")

    if semaphore is None:
        return await _call_with_retries()
    async with semaphore:
        return await _call_with_retries()


async def run_extraction_async(
    df: pd.DataFrame,
    few_shot_df: pd.DataFrame | None,
    max_concurrency: int,
    model: str,
    system_prompt: str,
    temperature: float,
    progress_bar,
    client,
) -> tuple[pd.DataFrame, dict]:
    """Returns (predictions_df, aggregated_usage_dict)."""
    few_shot_examples = few_shot_df.to_dict(orient="records") if few_shot_df is not None else None
    rows = df.to_dict(orient="records")
    sem = asyncio.Semaphore(max_concurrency)

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    async def _one(i, row):
        result, usage = await extract_row(
            row=row,
            few_shot_examples=few_shot_examples,
            model=model,
            semaphore=sem,
            system_prompt=system_prompt,
            temperature=temperature,
            client=client,
        )
        return i, result, usage

    tasks = [asyncio.create_task(_one(i, r)) for i, r in enumerate(rows)]
    results = [None] * len(rows)

    for future in asyncio.as_completed(tasks):
        i, result, usage = await future
        results[i] = result
        total_usage["prompt_tokens"] += usage["prompt_tokens"]
        total_usage["completion_tokens"] += usage["completion_tokens"]
        progress_bar.update(1)

    return pd.DataFrame(results), total_usage


def compute_column_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
            output_dict=True,
        ),
    }


def evaluate_run(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict[str, dict]:
    joined = pd.merge(gt_df, pred_df, on="row_id", how="inner", suffixes=("", "_pred"))
    n_expected, n_joined = len(gt_df), len(joined)
    if n_joined < n_expected:
        print(f"  WARNING: {n_expected - n_joined} rows lost during join")
    metrics: dict[str, dict] = {}
    for gt_col, pred_col in EVAL_COLUMNS.items():
        y_true = joined[gt_col].tolist()
        y_pred = joined[pred_col].tolist()
        metrics[gt_col] = compute_column_metrics(y_true, y_pred)
    return metrics


def _extract_first_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().lower()
    if not text:
        return None
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    number = float(match.group())
    if "ms" in text:
        return number
    if "sec" in text or text.endswith("s"):
        return number * 1000.0
    return number


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_vals[low]
    weight = rank - low
    return sorted_vals[low] * (1 - weight) + sorted_vals[high] * weight


def compute_hypothesis_metrics(
    *,
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    usage: dict[str, int],
    model: str | object,
    latency_ms: list[float] | None = None,
    risky_keywords: tuple[str, ...] = ("lawyer", "fire hazard", "spanish"),
) -> dict[str, float]:
    """
    Compute Module 3.3 hypothesis proxy metrics (H2/H3/H4/H6/H7).

    - H2 routing: accuracy on Category.
    - H3 drafts: normalized edit similarity (%), using [Human] Initial Response as gold.
    - H4 safety: escalation rate on risky-keyword subset.
    - H6 economics: total and per-row cost from token usage.
    - H7 latency: p90 in ms (from explicit timings or parsed latency column).
    """
    joined = pd.merge(gt_df, pred_df, on="row_id", how="inner", suffixes=("", "_pred"))
    row_count = len(joined)
    if row_count == 0:
        return {
            "row_count": 0.0,
            "h2_routing_accuracy": 0.0,
            "h3_draft_edit_similarity": 0.0,
            "h4_safety_escalation_rate": 0.0,
            "h4_risky_subset_size": 0.0,
            "h6_total_cost_usd": 0.0,
            "h6_cost_per_row_usd": 0.0,
            "h7_p90_latency_ms": 0.0,
        }

    h2_accuracy = float(
        accuracy_score(
            joined["Category"].tolist(),
            joined["category"].tolist(),
        )
    )

    gold_responses = joined["[Human] Initial Response"].fillna("").tolist()
    pred_responses = joined["[Agent] Initial Response"].fillna("").tolist()
    similarities = [
        SequenceMatcher(None, str(gold), str(pred)).ratio()
        for gold, pred in zip(gold_responses, pred_responses, strict=True)
    ]
    h3_similarity = float(sum(similarities) / len(similarities)) if similarities else 0.0

    requests = joined["Request Text"].fillna("").tolist()
    routed_dept = joined["[Agent] Routing to Department"].fillna("").tolist()
    risk_col = (
        joined["Risks (yes/no)"].fillna("").tolist()
        if "Risks (yes/no)" in joined.columns
        else [""] * row_count
    )
    risky_keywords_lower = tuple(k.lower() for k in risky_keywords)
    risky_idx = [
        i
        for i, req in enumerate(requests)
        if any(k in str(req).lower() for k in risky_keywords_lower)
    ]
    escalated_count = 0
    for i in risky_idx:
        explicit_risk = str(risk_col[i]).strip().lower() == "yes"
        routed_support = str(routed_dept[i]).strip().lower() == "customer support"
        if explicit_risk or routed_support:
            escalated_count += 1
    h4_escalation = float(escalated_count / len(risky_idx)) if risky_idx else 0.0

    total_cost = compute_cost(usage, model_id(model))
    cost_per_row = total_cost / row_count

    latencies = latency_ms or []
    if not latencies and "Latency" in joined.columns:
        parsed = [_extract_first_float(v) for v in joined["Latency"].tolist()]
        latencies = [v for v in parsed if v is not None]
    h7_p90 = _percentile(latencies, 0.9) if latencies else 0.0

    return {
        "row_count": float(row_count),
        "h2_routing_accuracy": h2_accuracy,
        "h3_draft_edit_similarity": h3_similarity,
        "h4_safety_escalation_rate": h4_escalation,
        "h4_risky_subset_size": float(len(risky_idx)),
        "h6_total_cost_usd": float(total_cost),
        "h6_cost_per_row_usd": float(cost_per_row),
        "h7_p90_latency_ms": float(h7_p90),
    }


def generate_report(
    *,
    phase1_agg: pd.DataFrame,
    df_phase2: pd.DataFrame,
    best_model: str,
    best_nshot: int,
    few_shot_counts: list[int],
    temperature: float,
) -> str:
    """Build a markdown summary report for notebook outputs."""
    lines: list[str] = []
    lines.append("# Model Evaluation Report")
    lines.append(f"**Date:** {datetime.now():%Y-%m-%d %H:%M}  ")
    lines.append(f"**Models tested (Phase 1):** {', '.join(short_name(m) for m in MODELS)}  ")
    lines.append(f"**Few-shot counts tested (Phase 2):** {few_shot_counts}  ")
    lines.append(f"**Temperature:** {temperature}")
    lines.append("")
    lines.append("## Phase 1 - Model Selection (0-shot + cost)")
    lines.append("")
    header = "| Model | Avg Accuracy | Avg F1 Macro | Cost (USD) | F1 / \\$ |"
    sep = "|---|---|---|---|---|"
    rows = []
    for row in phase1_agg.to_dict("records"):
        rows.append(
            f"| {row['model_short']} | {row['avg_accuracy']:.4f} | "
            f"{row['avg_f1_macro']:.4f} | \\${row['cost_usd']:.6f} | "
            f"{row['f1_per_dollar']:.2f} |"
        )
    lines.extend([header, sep] + rows)
    lines.append("")
    lines.append(f"**Selected:** {short_name(model_id(best_model))}")
    lines.append("")
    lines.append("## Phase 2 - Few-Shot Ablation")
    lines.append("")
    for gt_col in EVAL_COLUMNS:
        lines.append(f"### {gt_col}")
        lines.append("")
        sub = df_phase2[df_phase2["target_column"] == gt_col].sort_values("n_shot")
        lines.extend(
            [
                "| n-shot | Accuracy | F1 Macro | Cost (USD) |",
                "|---|---|---|---|",
                *[
                    (
                        f"| {r['n_shot']} | {r['accuracy']:.4f} | "
                        f"{r['f1_macro']:.4f} | \\${r['cost_usd']:.6f} |"
                    )
                    for r in sub.to_dict("records")
                ],
            ]
        )
        lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        f"**{short_name(model_id(best_model))}** with **{best_nshot}** few-shot examples "
        f"achieves the highest average F1-macro across both targets and offers "
        f"the best cost-efficiency ratio. Use this configuration for production."
    )
    return "\n".join(lines)
