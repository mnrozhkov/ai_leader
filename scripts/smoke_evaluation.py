#!/usr/bin/env python3
"""Offline smoke test for async evaluation + model comparison (FakeClient only).

Run from repo root:
  uv run python scripts/smoke_evaluation.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ai_leader.clients import FakeClient  # noqa: E402
from ai_leader.experiments import (  # noqa: E402
    build_model_comparison_dataframe,
    evaluate_model_on_dataframe_async,
    run_model_comparison_async,
)
from ai_leader.prompts import DEFAULT_SYSTEM_PROMPT  # noqa: E402


def _tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [0, 1],
            "Request Text": ["Hello", "Order issue"],
            "Submission Channel": ["Email", "Chat"],
            "Order History": [None, None],
            "Category": ["Payment", "Order Issue"],
            "Routing to Department": ["Customer Support", "Returns"],
            "[Human] Initial Response": ["Thanks", "We can help"],
        }
    )


async def main() -> None:
    df = _tiny_df()
    client = FakeClient()

    run = await evaluate_model_on_dataframe_async(
        df=df,
        model="smoke-model",
        client=client,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        temperature=0.0,
        max_concurrency=2,
        use_progress=False,
    )
    assert len(run.predictions) == 2, len(run.predictions)

    results = await run_model_comparison_async(
        df=df,
        models=["m1", "m2"],
        client=client,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        temperature=0.0,
        max_concurrency=2,
        use_progress=False,
    )
    assert set(results) == {"m1", "m2"}
    compact = build_model_comparison_dataframe(df, results)
    assert len(compact) == 2
    full = build_model_comparison_dataframe(df, results, show_all=True)
    assert "auto_route_coverage" in full.columns
    print("smoke_evaluation: ok")


if __name__ == "__main__":
    asyncio.run(main())
