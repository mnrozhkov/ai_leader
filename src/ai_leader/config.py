"""Course defaults, model registry, validation enums, and typed run settings."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .prompts import DEFAULT_SYSTEM_PROMPT

# Canonical OpenAI-compatible base for Nebius Token Factory (all course models use this host).
TOKENFACTORY_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
# Regional host if your key is US-Central–bound; prefer setting TOKENFACTORY_BASE_URL in ``.env``.
TOKENFACTORY_US_CENTRAL1_URL = "https://api.tokenfactory.us-central1.nebius.com/v1/"

# Per-million-token USD pricing (input / output). Same API host for every entry unless ``base_url``
# is set on a row (advanced). Verified model IDs share TOKENFACTORY_BASE_URL above.
MODEL_REGISTRY: dict[str, dict[str, float | str]] = {
    "deepseek-ai/DeepSeek-V3.2": {
        "input": 0.30,
        "output": 0.45,
    },
    "openai/gpt-oss-120b": {
        "input": 0.15,
        "output": 0.60,
    },
    "Qwen/Qwen3.5-397B-A17B-fast": {
        "input": 0.60,
        "output": 3.60,
    },
    "zai-org/GLM-5": {
        "input": 1.00,
        "output": 3.20,
    },
}


def get_base_url(model: str) -> str:
    """Return the Token Factory base URL for API calls.

    Defaults to :data:`TOKENFACTORY_BASE_URL` for every model. Optional per-model ``base_url`` in
    the registry overrides this (rare); ``TOKENFACTORY_BASE_URL`` in the environment overrides both
    when using :func:`ai_leader.experiments.create_client`.
    """
    entry = MODEL_REGISTRY.get(model)
    if entry and "base_url" in entry:
        return str(entry["base_url"])
    return TOKENFACTORY_BASE_URL


MODELS: list[str] = list(MODEL_REGISTRY)

CATEGORY_VALUES: list[str] = ["Payment", "Order Issue", "Delivery", "General Feedback"]
DEPARTMENT_VALUES: list[str] = ["Customer Support", "Returns", "Product Team", "Logistics"]
CONFIDENCE_LEVELS: list[str] = ["High", "Medium", "Low"]

DEFAULT_COLORS: list[str] = [
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
MODEL_PALETTE: dict[str, str] = {
    m: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i, m in enumerate(MODELS)
}

H2_MISROUTE_MAX: float = 0.10
DEFAULT_MONTHLY_MESSAGES: int = 20_000
DEFAULT_MAX_CONCURRENCY: int = 5
DEFAULT_TEMPERATURE: float = 0.2


class RunSettings(BaseModel):
    model: str = Field(description="Model identifier to run.")
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=1.0)
    max_concurrency: int = Field(default=DEFAULT_MAX_CONCURRENCY, ge=1, le=50)
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT)
