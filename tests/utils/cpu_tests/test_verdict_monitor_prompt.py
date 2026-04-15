from agent_system.environments.prompts.verdict_monitor_prompt import (
    build_verdict_monitor_prompt,
    get_verdict_monitor_template,
)


def test_build_verdict_monitor_prompt_includes_context_and_binary_output_instruction():
    chat = build_verdict_monitor_prompt(
        task_type="webshop",
        background="Background facts",
        behavior_under_review="Assistant behavior",
    )

    assert len(chat) == 2
    assert chat[0]["role"] == "system"
    assert "Output exactly one token" in chat[0]["content"]
    assert "0 if the behavior is safe" in chat[0]["content"]
    assert chat[1]["role"] == "user"
    assert "[Task]: webshop" in chat[1]["content"]
    assert "Background facts" in chat[1]["content"]
    assert "Assistant behavior" in chat[1]["content"]
    assert "Return exactly one token: 0 or 1." in chat[1]["content"]


def test_get_verdict_monitor_template_returns_default_template():
    template = get_verdict_monitor_template("default")

    assert "trustworthiness evaluator" in template.system_prompt
