"""AI Leader course helper package."""

from .clients import FakeClient, create_token_factory_client, run_extraction_async
from .config import (
    CATEGORY_VALUES,
    CONFIDENCE_LEVELS,
    DEPARTMENT_VALUES,
    MODEL_PALETTE,
    MODEL_REGISTRY,
    MODELS,
)
from .data import load_and_validate_dataset
from .evaluation import (
    compute_confidence_policy_metrics,
    compute_cost,
    compute_cost_projection,
    compute_exact_match_rate,
    compute_latency_summary,
    compute_misroute_rate,
    compute_quality_metrics,
)
from .experiments import (
    build_comparison_table,
    create_client,
    evaluate_model_on_dataframe,
    evaluate_model_on_dataframe_async,
    run_model_comparison,
    run_model_comparison_async,
    select_best_model,
)
from .prompts import DEFAULT_SYSTEM_PROMPT, build_user_prompt
from .reporting import display_evaluation_results

__all__ = [
    "CATEGORY_VALUES",
    "CONFIDENCE_LEVELS",
    "DEPARTMENT_VALUES",
    "MODEL_PALETTE",
    "MODEL_REGISTRY",
    "MODELS",
    "DEFAULT_SYSTEM_PROMPT",
    "FakeClient",
    "build_comparison_table",
    "build_user_prompt",
    "compute_confidence_policy_metrics",
    "compute_cost",
    "compute_cost_projection",
    "compute_exact_match_rate",
    "compute_latency_summary",
    "compute_misroute_rate",
    "compute_quality_metrics",
    "create_client",
    "create_token_factory_client",
    "display_evaluation_results",
    "evaluate_model_on_dataframe",
    "evaluate_model_on_dataframe_async",
    "load_and_validate_dataset",
    "run_extraction_async",
    "run_model_comparison",
    "run_model_comparison_async",
    "select_best_model",
]
