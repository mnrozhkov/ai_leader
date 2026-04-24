"""Model client adapters."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from openai import APIError, AsyncOpenAI

from .prompts import build_user_prompt
from .schemas import RESPONSE_SCHEMA, SupportTicketExtraction

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 30.0


def create_token_factory_client(
    api_key: str,
    *,
    base_url: str | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)


def _strip_markdown_json(content: str) -> str:
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return content


def _parse_model_output(content: str) -> dict[str, Any]:
    content = _strip_markdown_json(content.strip())
    parsed = json.loads(content)
    validated = SupportTicketExtraction.model_validate(parsed)
    return validated.model_dump(by_alias=True, mode="json")


async def _openai_completion(
    client: AsyncOpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> tuple[str, dict[str, int]]:
    completion = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=RESPONSE_SCHEMA,
    )
    content = (completion.choices[0].message.content or "").strip()
    if not content:
        raise ValueError(
            f"Model returned empty content. Finish reason: {completion.choices[0].finish_reason}"
        )
    usage = {
        "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(completion.usage, "completion_tokens", 0) or 0,
    }
    return content, usage


async def extract_row(
    *,
    row: dict[str, Any],
    few_shot_examples: list[dict] | None,
    model: str,
    semaphore: asyncio.Semaphore | None,
    system_prompt: str,
    temperature: float,
    client: AsyncOpenAI | FakeClient,
    max_retries: int = 3,
) -> tuple[dict[str, Any], dict[str, int], float]:
    user_prompt = build_user_prompt(
        request_text=row["Request Text"],
        submission_channel=row["Submission Channel"],
        order_history=row.get("Order History"),
        few_shot_examples=few_shot_examples,
    )

    async def _call() -> tuple[dict[str, Any], dict[str, int], float]:
        start = time.perf_counter()
        if isinstance(client, FakeClient):
            content, usage = await client.create_chat_completion(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                row=row,
            )
        else:
            content, usage = await _openai_completion(
                client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        result = _parse_model_output(content)
        result["row_id"] = row["row_id"]
        return result, usage, latency_ms

    async def _call_with_retries() -> tuple[dict[str, Any], dict[str, int], float]:
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                return await asyncio.wait_for(
                    _call(),
                    timeout=_DEFAULT_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                log.warning(
                    "Row %s skipped — TimeoutError after %.0fs",
                    row.get("row_id", "?"),
                    _DEFAULT_TIMEOUT_SECONDS,
                )
                raise
            except APIError as exc:
                log.warning(
                    "Row %s skipped — %s: %s",
                    row.get("row_id", "?"),
                    type(exc).__name__,
                    exc,
                )
                raise
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == max_retries:
                    raise
                delay = 2**attempt
                log.warning(
                    "Row %s attempt %d/%d failed (%s), retrying in %ds",
                    row.get("row_id", "?"),
                    attempt,
                    max_retries,
                    type(exc).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError("Unreachable retry state") from last_error

    if semaphore is None:
        return await _call_with_retries()
    async with semaphore:
        return await _call_with_retries()


async def run_extraction_async(
    *,
    df: pd.DataFrame,
    few_shot_df: pd.DataFrame | None,
    max_concurrency: int,
    model: str,
    system_prompt: str,
    temperature: float,
    progress_bar: Any | None,
    client: AsyncOpenAI | FakeClient,
) -> tuple[pd.DataFrame, dict[str, int], list[float]]:
    few_shot_examples = few_shot_df.to_dict(orient="records") if few_shot_df is not None else None
    rows = df.to_dict(orient="records")
    sem = asyncio.Semaphore(max_concurrency)

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    latency_ms: list[float] = []

    async def _one(
        i: int,
        row: dict[str, Any],
    ) -> tuple[int, dict[str, Any] | None, dict[str, int], float]:
        try:
            result, usage, latency = await extract_row(
                row=row,
                few_shot_examples=few_shot_examples,
                model=model,
                semaphore=sem,
                system_prompt=system_prompt,
                temperature=temperature,
                client=client,
            )
            return i, result, usage, latency
        except Exception as exc:
            log.error(
                "Row %s failed after retries (%s: %s) — skipping",
                row.get("row_id", i),
                type(exc).__name__,
                exc,
            )
            return i, None, {"prompt_tokens": 0, "completion_tokens": 0}, 0.0

    tasks = [asyncio.create_task(_one(i, r)) for i, r in enumerate(rows)]
    results: list[dict[str, Any] | None] = [None] * len(rows)
    failed = 0

    for future in asyncio.as_completed(tasks):
        i, result, usage, latency = await future
        results[i] = result
        if result is None:
            failed += 1
        total_usage["prompt_tokens"] += usage["prompt_tokens"]
        total_usage["completion_tokens"] += usage["completion_tokens"]
        latency_ms.append(latency)
        if progress_bar is not None:
            progress_bar.update(1)

    if failed:
        log.warning("Skipped %d / %d rows due to errors", failed, len(rows))

    return pd.DataFrame([r for r in results if r is not None]), total_usage, latency_ms


@dataclass(slots=True)
class FakeClient:
    default_category: str = "General Feedback"
    default_department: str = "Customer Support"
    confidence: str = "High"

    async def create_chat_completion(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        row: dict[str, Any],
    ) -> tuple[str, dict[str, int]]:
        category = row.get("Category", self.default_category)
        department = row.get("Routing to Department", self.default_department)
        response_text = row.get("[Human] Initial Response", "Thanks for reaching out.")
        payload = {
            "Category": category,
            "Routing to Department": department,
            "[Agent] Initial Response": response_text,
            "Confidence": row.get("Confidence", self.confidence),
        }
        usage = {"prompt_tokens": 10, "completion_tokens": 20}
        return json.dumps(payload, ensure_ascii=False), usage
