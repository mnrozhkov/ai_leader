# AI Leader

Course materials, notebooks, and exercises.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python 3.12+)
- Git

## Quick setup

```bash
# Install Python, runtime libs, and local dev tools (Jupyter, Ruff, pre-commit, etc.)
uv sync --extra dev

# Optional: pin the same Python as the repo
uv python install 3.12

# Install git hooks (secrets scan, Ruff lint + format, basic file checks)
uv run pre-commit install

# Run hooks on all files once (optional)
uv run pre-commit run --all-files
```

## JupyterLab

Open notebooks in the browser with the same environment as `uv sync`:

```bash
cd /path/to/ai-leader
uv run jupyter lab
```

Jupyter opens a local URL (copy the token from the terminal if prompted). In **Kernel → Change kernel**, pick the interpreter from this project’s `.venv` if it is not already selected.

Work inside the virtual environment when running tools:

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
ruff check .
ruff format .
basedpyright .
```

Or without activating:

```bash
uv run ruff check .
uv run ruff format .
uv run basedpyright .
```

## Colab quick start

```python
%pip install -q git+https://github.com/mnrozhkov/ai_leader.git
```

The default package dependencies are **runtime-only** (API, pandas, plotting, CLI) so Colab’s pinned `ipython` / `ipykernel` / `jupyter-*` stack is not upgraded. You may still see harmless pip warnings from other Colab preinstalls; the install should succeed.

Open notebooks directly in Colab:

- [Notebook A](https://colab.research.google.com/github/mnrozhkov/ai_leader/blob/main/notebooks/notebook_a.ipynb)
- [Notebook B](https://colab.research.google.com/github/mnrozhkov/ai_leader/blob/main/notebooks/notebook_b.ipynb)

## Local notebook development

If you want to test notebook changes before pushing:

```bash
cd /path/to/ai-leader
uv sync --extra dev
uv run jupyter lab
```

In the notebook, switch the install cell to:

```python
%pip install -q -e .
```

This keeps the notebook using your local code without a GitHub push.

## What pre-commit does

- **Secrets:** [Gitleaks](https://github.com/gitleaks/gitleaks) plus `detect-private-key` for accidental keys in commits
- **Lint / format:** [Ruff](https://docs.astral.sh/ruff/) (`ruff check`, `ruff format`)
- **Hygiene:** trailing whitespace, YAML/TOML/JSON checks, merge conflict markers, large files

## Project layout

- `pyproject.toml` — metadata and tool config (`ruff`, `basedpyright`, `uv`)
- `notebooks/3_3_run_evaluations_quality_safety_cost.ipynb` — Module 3.3 notebook
- `src/ai_leader/` — reusable notebook logic (data, prompts, evaluation, experiments, reporting)
- `scripts/run_notebook_ci.py` — CI notebook smoke runner
