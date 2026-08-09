from ai_leader.prompts import DEFAULT_SYSTEM_PROMPT, build_user_prompt
from ai_leader.schemas import SupportTicketExtraction


def test_build_user_prompt_includes_fields():
    prompt = build_user_prompt(
        request_text="Help with payment",
        submission_channel="Email",
        order_history=None,
        few_shot_examples=None,
    )
    assert "Request Text" in prompt
    assert "Submission Channel" in prompt
    assert "Order History" in prompt


def test_routing_output_contract_has_only_scored_fields() -> None:
    schema = SupportTicketExtraction.model_json_schema()

    assert set(schema["properties"]) == {"Category", "Routing to Department", "Confidence"}
    assert set(schema["required"]) == {"Category", "Routing to Department", "Confidence"}
    assert "[Agent] Initial Response" not in DEFAULT_SYSTEM_PROMPT
