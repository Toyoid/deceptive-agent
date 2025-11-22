import pytest

from agent_system.environments.env_package.reason_chat.projection import (
    reason_chat_projection,
)


@pytest.mark.parametrize(
    "action, expected_thought, expected_answer, expected_valid",
    [
        (
            "<think> deliberate plan </think><answer>final output</answer>",
            "deliberate plan",
            "final output",
            1,
        ),
        (
            "\n <Think>\n  multi-line\nreasoning \n</THINK>\n<Answer>\n Detailed\n</AnSwEr>\n",
            "multi-line\nreasoning",
            "Detailed",
            1,
        ),
        (
            "<think>Reasoning only</think>\nHere is the final answer.",
            "Reasoning only",
            "Here is the final answer.",
            1,
        ),
        (
            "Answer without think block.",
            "",
            "Answer without think block.",
            0,
        ),
        (
            "lead-in text<think>plan</think><answer>reply</answer>",
            "plan",
            "reply",
            0,
        ),
        (
            "<think>first</think> extra <think>second</think><answer>reply</answer>",
            "first",
            "reply",
            0,
        ),
        (
            "<think>plan</think><answer>first</answer><answer>second</answer>",
            "plan",
            "first",
            0,
        ),
        (
            "<think>plan</think><answer>dangling",
            "plan",
            "<answer>dangling",
            0,
        ),
        (
            "<think>plan</think>final text</answer>",
            "plan",
            "final text</answer>",
            0,
        ),
    ],
)
def test_reason_chat_projection_various_cases(
    action: str, expected_thought: str, expected_answer: str, expected_valid: int
) -> None:
    projected, valids = reason_chat_projection([action])
    assert projected == [
        {"reason": expected_thought, "answer": expected_answer, "raw_text": action}
    ]
    assert valids == [expected_valid]


def test_reason_chat_projection_batch_order_preserved() -> None:
    actions = [
        "<think>a</think><answer>first</answer>",
        "<think>b</think>second",
        "missing think but has answer",
    ]

    projected, valids = reason_chat_projection(actions)

    assert [item["reason"] for item in projected] == ["a", "b", ""]
    assert [item["answer"] for item in projected] == [
        "first",
        "second",
        "missing think but has answer",
    ]
    assert [item["raw_text"] for item in projected] == actions
    assert valids == [1, 1, 0]
