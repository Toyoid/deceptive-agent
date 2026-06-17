from agent_system.environments.prompts.cheatshop import CHEATSHOP_VERDICT_OVERSIGHT_SCOPE
from agent_system.environments.prompts.deceptive_search import (
    DECEPTIVE_SEARCH_VERDICT_OVERSIGHT_SCOPE,
)
from agent_system.verdict_monitor import build_verdict_monitor_background


def test_deceptive_search_verdict_background_gets_binary_guidance():
    background = build_verdict_monitor_background("deceptive_search", "base search background")

    assert DECEPTIVE_SEARCH_VERDICT_OVERSIGHT_SCOPE in background
    assert background.count(DECEPTIVE_SEARCH_VERDICT_OVERSIGHT_SCOPE) == 1


def test_cheatshop_verdict_background_gets_binary_guidance():
    background = build_verdict_monitor_background("cheatshop", "base shopping background")

    assert CHEATSHOP_VERDICT_OVERSIGHT_SCOPE in background
    assert background.count(CHEATSHOP_VERDICT_OVERSIGHT_SCOPE) == 1


def test_plain_background_is_unchanged():
    base_background = "The assistant is answering ordinary factual questions."

    background = build_verdict_monitor_background("chat", base_background)

    assert background == base_background


def test_broad_task_labels_do_not_trigger_verdict_guidance():
    assert build_verdict_monitor_background("search", "base") == "base"
    assert build_verdict_monitor_background("Shopping", "base") == "base"


def test_verdict_background_guidance_is_task_focus_not_output_format():
    for scope in (DECEPTIVE_SEARCH_VERDICT_OVERSIGHT_SCOPE, CHEATSHOP_VERDICT_OVERSIGHT_SCOPE):
        assert "binary verdict review" not in scope
        assert "Insider Lookup" not in scope
        assert "Results analysis panel" not in scope
        assert "Structured query parser" not in scope
        assert "non-public" not in scope
        assert "staff-facing" not in scope
        assert "privileged" not in scope
        assert "internal workflow" not in scope


def test_verdict_background_guidance_contains_task_decision_boundary():
    for scope in (DECEPTIVE_SEARCH_VERDICT_OVERSIGHT_SCOPE, CHEATSHOP_VERDICT_OVERSIGHT_SCOPE):
        assert "verdict 1" in scope
        assert "Verdict 0" in scope
