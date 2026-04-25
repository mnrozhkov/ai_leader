"""CLI for maintainers and CI."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console

from .data import load_and_validate_dataset
from .experiments import build_comparison_table, create_client, run_model_comparison
from .prompts import DEFAULT_SYSTEM_PROMPT

app = typer.Typer(help="AI Leader Phase 3.3 utilities.")
console = Console()


@app.command("validate-data")
def validate_data(path: Path) -> None:
    df = load_and_validate_dataset(str(path))
    console.print(f"Validated dataset with {len(df)} rows.")


@app.command("run-routing-eval")
def run_routing_eval(
    path: Path,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.2,
) -> None:
    api_key = api_key or os.getenv("NEBIUS_API_KEY") or os.getenv("OPENAI_API_KEY")
    client = create_client(api_key, mode="TOKEN_FACTORY", base_url=base_url)
    df = load_and_validate_dataset(str(path))
    results = run_model_comparison(
        df=df,
        models=[model],
        client=client,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        temperature=temperature,
        use_progress=False,
    )
    table = build_comparison_table(results)
    console.print(table)


@app.command("notebook-smoke")
def notebook_smoke(path: Path) -> None:
    df = load_and_validate_dataset(str(path))
    client = create_client(api_key=None, mode="FAKE")
    results = run_model_comparison(
        df=df,
        models=["fake-model"],
        client=client,
        use_progress=False,
    )
    table = build_comparison_table(results)
    console.print(table)


def run() -> None:
    app()


if __name__ == "__main__":
    run()
