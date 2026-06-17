from __future__ import annotations

from agent_system.environments.prompts.cheatshop import CHEATSHOP_VERDICT_OVERSIGHT_SCOPE
from agent_system.environments.prompts.deceptive_search import DECEPTIVE_SEARCH_VERDICT_OVERSIGHT_SCOPE


def _append_once(background: str, note: str) -> str:
    text = str(background)
    if note in text:
        return text
    return f"{text.rstrip()}\n\n{note}" if text.strip() else note


def build_verdict_monitor_background(task_type: str, monitor_background: str) -> str:
    """Add task-specific binary-verdict guidance without inspecting prompt contents."""
    background = str(monitor_background)
    normalized_task_type = str(task_type).strip().lower()
    if normalized_task_type == "deceptive_search":
        return _append_once(background, DECEPTIVE_SEARCH_VERDICT_OVERSIGHT_SCOPE)
    if normalized_task_type == "cheatshop":
        return _append_once(background, CHEATSHOP_VERDICT_OVERSIGHT_SCOPE)
    return background
