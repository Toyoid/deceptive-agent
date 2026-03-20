# Copyright 2026 Hanxiao Li, Beihang University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from agent_system.environments.env_package.deceptive_search.projection import (
    INVALID_FINAL_ANSWER,
    INVALID_FINAL_ANSWER_TOOL,
    extract_search_final_resps,
)


@pytest.mark.parametrize(
    "action, expected",
    [
        (
            "<think>plan carefully</think><answer>Paris</answer><source>[1] states the capital directly.</source>",
            {
                "thought": "<think>\nplan carefully\n</think>",
                "answer": "<answer>\nParis\n</answer>\n<source>\n[1] states the capital directly.\n</source>",
            },
        ),
        (
            "<think>need more evidence</think><search>capital of france</search>",
            {
                "thought": "<think>\nneed more evidence\n</think>",
                "answer": f"<answer>\n{INVALID_FINAL_ANSWER_TOOL.format(tool_call_tag='<search>')}\n</answer>\n<source>\n</source>",
            },
        ),
        (
            "<think>use insider</think><insider>capital of france</insider>",
            {
                "thought": "<think>\nuse insider\n</think>",
                "answer": f"<answer>\n{INVALID_FINAL_ANSWER_TOOL.format(tool_call_tag='<insider>')}\n</answer>\n<source>\n</source>",
            },
        ),
        (
            "Plain text without any XML tags.",
            {
                "thought": "<think>\n</think>",
                "answer": f"<answer>\n{INVALID_FINAL_ANSWER}\n</answer>\n<source>\n</source>",
            },
        ),
        (
            "<think>enough evidence</think><answer>Paris</answer>",
            {
                "thought": "<think>\nenough evidence\n</think>",
                "answer": "<answer>\nParis\n</answer>\n<source>\n</source>",
            },
        ),
        (
            "<think>mixed output</think><search>capital of france</search><answer>Paris</answer><source>[1]</source>",
            {
                "thought": "<think>\nmixed output\n</think>",
                "answer": f"<answer>\n{INVALID_FINAL_ANSWER_TOOL.format(tool_call_tag='<search>')}\n</answer>\n<source>\n</source>",
            },
        ),
    ],
)
def test_extract_search_final_resps_various_cases(
    action: str, expected: dict[str, str]
) -> None:
    assert extract_search_final_resps([action]) == [expected]


def test_extract_search_final_resps_preserves_batch_order() -> None:
    actions = [
        "<think>a</think><answer>first</answer><source>[1]</source>",
        "<think>b</think><search>lookup</search>",
        "<answer>third</answer>",
    ]

    projected = extract_search_final_resps(actions)

    assert [item["thought"] for item in projected] == [
        "<think>\na\n</think>",
        "<think>\nb\n</think>",
        "<think>\n</think>",
    ]
    assert [item["answer"] for item in projected] == [
        "<answer>\nfirst\n</answer>\n<source>\n[1]\n</source>",
        f"<answer>\n{INVALID_FINAL_ANSWER_TOOL.format(tool_call_tag='<search>')}\n</answer>\n<source>\n</source>",
        "<answer>\nthird\n</answer>\n<source>\n</source>",
    ]
