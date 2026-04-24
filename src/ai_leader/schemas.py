"""Pydantic models and schema literals."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from .config import CATEGORY_VALUES, CONFIDENCE_LEVELS, DEPARTMENT_VALUES

Category = Literal["Payment", "Order Issue", "Delivery", "General Feedback"]
Department = Literal["Customer Support", "Returns", "Product Team", "Logistics"]
ConfidenceLevel = Literal["High", "Medium", "Low"]


class SupportTicketExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Category = Field(
        description="Best-fit category for the customer request.",
        validation_alias=AliasChoices("Category", "category"),
        serialization_alias="category",
    )
    routing_to_department: Department = Field(
        description="Department that should handle the request.",
        validation_alias=AliasChoices("Routing to Department", "[Agent] Routing to Department"),
        serialization_alias="[Agent] Routing to Department",
    )
    agent_initial_response: str = Field(
        description="A professional first response to the customer.",
        validation_alias=AliasChoices("[Agent] Initial Response", "agent_initial_response"),
        serialization_alias="[Agent] Initial Response",
    )
    confidence: ConfidenceLevel | None = Field(
        default=None,
        description="Confidence for the routing decision.",
        validation_alias=AliasChoices("Confidence", "confidence"),
        serialization_alias="Confidence",
    )


class DatasetRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    request_text: str = Field(alias="Request Text")
    submission_channel: str = Field(alias="Submission Channel")
    category: Category = Field(alias="Category")
    routing_to_department: Department = Field(alias="Routing to Department")


class ModelUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ConfidencePolicyMetrics(BaseModel):
    high_confidence_coverage: float
    high_confidence_error_rate: float
    manual_review_rate: float


class CostProjection(BaseModel):
    cost_per_message_usd: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    monthly_cost_usd: float
    annual_cost_usd: float
    cost_per_correct_route_usd: float | None


class LatencySummary(BaseModel):
    median_latency_ms: float
    p95_latency_ms: float
    mean_latency_ms: float


RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "support_ticket_extraction",
        "schema": SupportTicketExtraction.model_json_schema(),
        "strict": True,
    },
}


def is_valid_category(value: str) -> bool:
    return value in CATEGORY_VALUES


def is_valid_department(value: str) -> bool:
    return value in DEPARTMENT_VALUES


def is_valid_confidence(value: str) -> bool:
    return value in CONFIDENCE_LEVELS
