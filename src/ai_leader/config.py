"""Course defaults, model registry, validation enums, and typed run settings."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .prompts import DEFAULT_SYSTEM_PROMPT

MODEL_REGISTRY: dict[str, dict[str, float]] = {
    "deepseek-ai/DeepSeek-V3.2": {"input": 0.30, "output": 0.45},
    "zai-org/GLM-5": {"input": 1.00, "output": 3.20},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
}

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
DEFAULT_MAX_CONCURRENCY: int = 6
DEFAULT_TEMPERATURE: float = 0.2


class RunSettings(BaseModel):
    model: str = Field(description="Model identifier to run.")
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=1.0)
    max_concurrency: int = Field(default=DEFAULT_MAX_CONCURRENCY, ge=1, le=50)
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT)
