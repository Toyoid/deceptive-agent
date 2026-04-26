"""Actor-only API rollout evaluation for agent environments."""

__all__ = [
    "ApiRolloutResult",
    "ApiRolloutRunner",
    "ChatResponse",
    "EvalRow",
    "OpenAICompatibleChatClient",
    "load_eval_rows",
]


def __getattr__(name):
    if name in {"ChatResponse", "OpenAICompatibleChatClient"}:
        from .clients import ChatResponse, OpenAICompatibleChatClient

        return {"ChatResponse": ChatResponse, "OpenAICompatibleChatClient": OpenAICompatibleChatClient}[name]
    if name in {"EvalRow", "load_eval_rows"}:
        from .data import EvalRow, load_eval_rows

        return {"EvalRow": EvalRow, "load_eval_rows": load_eval_rows}[name]
    if name in {"ApiRolloutResult", "ApiRolloutRunner"}:
        from .runner import ApiRolloutResult, ApiRolloutRunner

        return {"ApiRolloutResult": ApiRolloutResult, "ApiRolloutRunner": ApiRolloutRunner}[name]
    raise AttributeError(name)
