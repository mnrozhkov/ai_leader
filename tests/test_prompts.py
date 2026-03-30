from ai_leader.prompts import build_user_prompt


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
