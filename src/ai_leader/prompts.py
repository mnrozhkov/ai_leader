"""Prompt construction helpers."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

DEFAULT_SYSTEM_PROMPT = """You are a support routing assistant.
Return a JSON object with:
- Category
- Routing to Department
- [Agent] Initial Response
- Confidence (High, Medium, Low)
Follow the schema exactly and do not add extra keys.
""".strip()

IMPROVED_SYSTEM_PROMPT = """You are the AI routing assistant for Candlekeep, an e-commerce company \
that receives 10,000+ customer messages per month across email, chat, phone, and social channels.

Your task: classify each customer message and route it to the correct department.

## Categories
- **Payment** — refunds, charges, billing disputes, payment method updates
- **Order Issue** — damaged items, missing parts, wrong items, exchanges, returns
- **Delivery** — shipping delays, address changes, tracking, delivery window issues
- **General Feedback** — product praise/complaints, suggestions, bulk-order inquiries, \
non-order-specific questions

## Departments and routing rules
- **Customer Support** — handles Payment issues (refunds, billing, charges)
- **Returns** — handles Order Issue cases (damaged, missing parts, exchanges, wrong items)
- **Logistics** — handles Delivery problems (shipping, tracking, address changes)
- **Product Team** — handles General Feedback (product feedback, feature requests, suggestions)

## Key distinctions
- A billing complaint about a *charge* goes to Customer Support, not Returns.
- A complaint about a *damaged item* goes to Returns, not Customer Support.
- "Bulk order pricing" is General Feedback → Product Team, not Payment → Customer Support.
- Delivery-window complaints go to Logistics even if an order is involved.

## Confidence
- **High** — the category and department are clearly indicated by the message
- **Medium** — plausible but the message is ambiguous or could fit multiple categories
- **Low** — very uncertain; the message is vague or unusual

Return a JSON object with exactly these keys:
- Category
- Routing to Department
- [Agent] Initial Response
- Confidence (High, Medium, Low)

Do not add extra keys.
""".strip()


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
            output: dict[str, object] = {
                "Category": ex["Category"],
                "Routing to Department": ex["Routing to Department"],
                "[Agent] Initial Response": ex.get(
                    "[Agent] Initial Response",
                    ex.get("[Human] Initial Response", ""),
                ),
            }
            if "Confidence" in ex and ex["Confidence"]:
                output["Confidence"] = ex["Confidence"]
            parts.append(
                f"Example {i}\nInput:\n"
                f"  Request Text: {ex['Request Text']}\n"
                f"  Submission Channel: {ex['Submission Channel']}\n"
                f"  Order History: {ex.get('Order History')}\n"
                f"Output:\n{json.dumps(output, ensure_ascii=False)}"
            )
    parts.append("### New case — classify and respond ###")
    parts.append(f"Request Text: {request_text}")
    parts.append(f"Submission Channel: {submission_channel}")
    parts.append(f"Order History: {order_history if order_history is not None else 'null'}")
    return "\n\n".join(parts)


def _first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def _strip_markdown_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1]
        if value.endswith("```"):
            value = value[: -len("```")]
    return value.strip()


async def prepare_mistake_examples(
    *,
    eval_df: pd.DataFrame,
    predictions: pd.DataFrame,
    max_per_type: int = 3,
) -> pd.DataFrame:
    """Collect category/department mistake examples for prompt refinement."""
    if predictions.empty:
        return pd.DataFrame()

    joined = eval_df.copy()
    if "row_id" in predictions.columns:
        joined = joined.merge(predictions, on="row_id", how="inner")
    elif len(eval_df) == len(predictions):
        joined = joined.join(predictions, rsuffix="_pred")
    else:
        return pd.DataFrame()

    cols = joined.columns.tolist()
    pred_category_col = _first_existing_column(cols, ["category", "Category_pred", "Category"])
    pred_department_col = _first_existing_column(
        cols,
        [
            "[Agent] Routing to Department",
            "Routed to Department",
            "Routing to Department",
            "Routing to Department_pred",
            "routing_to_department",
        ],
    )
    if pred_category_col is None or pred_department_col is None:
        return pd.DataFrame()

    category_errors = joined[
        joined["Category"].astype(str) != joined[pred_category_col].astype(str)
    ].head(max_per_type)
    department_errors = joined[
        joined["Routing to Department"].astype(str) != joined[pred_department_col].astype(str)
    ].head(max_per_type)

    mistake_examples: list[dict[str, str]] = []
    for _, row in category_errors.iterrows():
        mistake_examples.append(
            {
                "type": "category",
                "request_text": str(row["Request Text"]),
                "gold_category": str(row["Category"]),
                "predicted_category": str(row[pred_category_col]),
                "gold_department": str(row["Routing to Department"]),
                "predicted_department": str(row[pred_department_col]),
            }
        )
    for _, row in department_errors.iterrows():
        mistake_examples.append(
            {
                "type": "department",
                "request_text": str(row["Request Text"]),
                "gold_category": str(row["Category"]),
                "predicted_category": str(row[pred_category_col]),
                "gold_department": str(row["Routing to Department"]),
                "predicted_department": str(row[pred_department_col]),
            }
        )

    if not mistake_examples:
        return pd.DataFrame()

    return pd.DataFrame(mistake_examples)


async def generate_prompt_from_mistakes(
    *,
    current_prompt: str,
    mistake_examples: pd.DataFrame | list[dict[str, str]],
    client: Any,
    model: str,
    temperature: float = 0.0,
) -> str:
    """Ask the model to revise the routing prompt from mistake examples."""
    if isinstance(mistake_examples, pd.DataFrame):
        if mistake_examples.empty:
            return current_prompt
        mistakes_payload = mistake_examples.to_dict(orient="records")
    else:
        if not mistake_examples:
            return current_prompt
        mistakes_payload = mistake_examples

    system_msg = (
        "You are an expert prompt engineer for customer-support routing. "
        "Return only the revised system prompt text. No markdown, no commentary."
    )
    user_msg = (
        "Current system prompt:\n"
        f"{current_prompt}\n\n"
        "Routing mistakes (JSON):\n"
        f"{json.dumps(mistakes_payload, ensure_ascii=False, indent=2)}\n\n"
        "Task:\n"
        "Revise the system prompt to reduce these mistakes while preserving "
        "strict JSON output requirements."
    )

    completion = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    revised_prompt = _strip_markdown_fence(completion.choices[0].message.content or "")
    return revised_prompt or current_prompt
