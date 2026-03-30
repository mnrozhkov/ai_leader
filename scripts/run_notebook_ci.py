from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def run_notebook(notebook_path: Path, fixture_path: Path) -> None:
    print(f"[notebook-ci] Using notebook: {notebook_path}")
    print(f"[notebook-ci] Using fixture: {fixture_path}")
    os.environ["AI_LEADER_CLIENT"] = "FAKE"
    os.environ["AI_LEADER_DATASET_PATH"] = str(fixture_path)
    print("[notebook-ci] Set AI_LEADER_FAKE_CLIENT=1")
    print(f"[notebook-ci] Set AI_LEADER_DATASET_PATH={fixture_path}")

    notebook = nbformat.read(notebook_path, as_version=4)
    print(f"[notebook-ci] Loaded notebook with {len(notebook.cells)} cells")
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    print("[notebook-ci] Starting execution")
    try:
        client.execute()
    except Exception as exc:
        print(f"[notebook-ci] Execution failed: {exc}")
        raise
    print("[notebook-ci] Execution finished successfully")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "notebooks" / "3_3_run_evaluations_quality_safety_cost.ipynb"
    fixture_path = root / "tests" / "fixtures" / "routing_eval_sample_small.csv"
    print(f"[notebook-ci] Repo root: {root}")
    run_notebook(notebook_path, fixture_path)


if __name__ == "__main__":
    main()
