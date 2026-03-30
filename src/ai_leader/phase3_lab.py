"""Legacy re-exports for the Phase 3 notebook lab."""

from __future__ import annotations

from .clients import run_extraction_async
from .config import MODEL_PALETTE, MODEL_REGISTRY, MODELS
from .evaluation import compute_cost, compute_quality_metrics
from .experiments import run_model_comparison, run_model_comparison_async, select_best_model
from .prompts import build_user_prompt

__all__ = [
    "MODEL_PALETTE",
    "MODEL_REGISTRY",
    "MODELS",
    "build_user_prompt",
    "compute_cost",
    "compute_quality_metrics",
    "run_extraction_async",
    "run_model_comparison",
    "run_model_comparison_async",
    "select_best_model",
]
