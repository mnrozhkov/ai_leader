"""Helpers for reviewing high-confidence routing errors."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

_ALLOWED_DEPARTMENTS = ["Logistics", "Customer Support", "Product Team", "Returns"]
_ALLOWED_CATEGORIES = ["Delivery", "Order Issue", "General Feedback", "Payment"]


def analyze_high_confident_errors(high_conf_errors_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize high-confidence error rows for downstream LLM analysis.

    Accepts either raw merged columns (e.g. ``Routing to Department`` / ``category``)
    or display-friendly columns (e.g. ``Department`` / ``Predicted Category``).
    """
    if high_conf_errors_df.empty:
        return pd.DataFrame(
            columns=[
                "row_id",
                "request_text",
                "category",
                "department",
                "predicted_category",
                "predicted_department",
                "confidence",
            ]
        )

    out = high_conf_errors_df.copy()
    rename_map = {
        "Request Text": "request_text",
        "Category": "category",
        "Routing to Department": "department",
        "Department": "department",
        "category": "predicted_category",
        "Predicted Category": "predicted_category",
        "[Agent] Routing to Department": "predicted_department",
        "Predicted Department": "predicted_department",
        "Confidence": "confidence",
    }
    out = out.rename(columns=rename_map)

    expected = [
        "row_id",
        "request_text",
        "category",
        "department",
        "predicted_category",
        "predicted_department",
        "confidence",
    ]
    present = [c for c in expected if c in out.columns]
    normalized = out[present].copy()
    for col in expected:
        if col not in normalized.columns:
            normalized[col] = ""
    return normalized[expected]


async def recommend_new_labels(
    analyzed_errors_df: pd.DataFrame,
    *,
    client: Any,
    model: str,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Run LLM label-review recommendations over analyzed high-confidence errors."""
    if analyzed_errors_df.empty:
        return pd.DataFrame(
            columns=[
                "row_id",
                "Original request",
                "category",
                "department",
                "recommendation",
                "Revise labels",
            ]
        )

    if max_rows is None:
        rows_df = analyzed_errors_df.copy()
    else:
        rows_df = analyzed_errors_df.head(max_rows).copy()
    llm_input: list[dict[str, Any]] = []
    for _, row in rows_df.iterrows():
        llm_input.append(
            {
                "row_id": int(row["row_id"]),
                "request_text": str(row["request_text"]),
                "category": str(row["category"]),
                "department": str(row["department"]),
                "predicted_category": str(row["predicted_category"]),
                "predicted_department": str(row["predicted_department"]),
                "confidence": str(row["confidence"]),
            }
        )

    analysis_prompt = (
        "You are reviewing labeling quality for support-ticket routing. "
        "Given one item, decide whether the existing reference label should change in category, "
        "department, both, or neither. "
        "Return strict JSON with keys: row_id (int), recommendation (string). "
        "Recommendation format MUST be one of: "
        "'Change category to <value>: <why>', "
        "'Change department to <value>: <why>', "
        "'Change category to <value> and department to <value>: <why>', "
        "or 'Change neither: <why>'. "
        f"Allowed Department values: {', '.join(_ALLOWED_DEPARTMENTS)}. "
        f"Allowed Category values: {', '.join(_ALLOWED_CATEGORIES)}. "
        "If a suggested value is outside allowed lists, use 'Change neither: <why>'."
    )

    recommendations: list[dict[str, str | int | bool]] = []
    pbar = tqdm(total=len(llm_input), desc="LLM relabel analysis", unit="item", leave=True)
    for item in llm_input:
        completion = await client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": analysis_prompt},
                {"role": "user", "content": json.dumps(item, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        payload = json.loads((completion.choices[0].message.content or "{}").strip())
        rec = str(payload.get("recommendation", "")).strip()
        if not rec:
            rec = "Change neither: insufficient evidence to justify re-labeling."
        if not rec.startswith("Change"):
            rec = f"Change {rec[0].lower() + rec[1:]}"
        if ":" not in rec:
            rec = (
                "Change neither: insufficient evidence to justify re-labeling."
                if rec.startswith("Change neither")
                else f"{rec}: rationale not provided by model."
            )

        allowed_dept_str = "|".join(_ALLOWED_DEPARTMENTS)
        allowed_cat_str = "|".join(_ALLOWED_CATEGORIES)
        has_invalid_department = "Change department to " in rec and not re.search(
            rf"Change department to ({allowed_dept_str})", rec
        )
        has_invalid_category = "Change category to " in rec and not re.search(
            rf"Change category to ({allowed_cat_str})", rec
        )
        if has_invalid_department or has_invalid_category:
            rec = "Change neither: proposed value was outside allowed category/department lists."

        recommendations.append(
            {
                "row_id": int(item["row_id"]),
                "Original request": str(item["request_text"]),
                "category": str(item["category"]),
                "department": str(item["department"]),
                "recommendation": rec,
            }
        )
        pbar.update(1)
    pbar.close()

    out = pd.DataFrame(recommendations)
    out["Revise labels"] = ~out["recommendation"].str.startswith("Change neither", na=False)
    return out[
        ["row_id", "Original request", "category", "department", "recommendation", "Revise labels"]
    ]
