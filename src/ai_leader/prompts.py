"""Prompt construction helpers."""

from __future__ import annotations

import json

DEFAULT_SYSTEM_PROMPT = """You are a support routing assistant.
Return a JSON object with:
- Category
- Routing to Department
- [Agent] Initial Response
- Confidence (High, Medium, Low)
Follow the schema exactly and do not add extra keys.
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
