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

from agent_system.environments.env_package.cheatshop.projection import cheatshop_projection


def test_cheatshop_projection_accepts_valid_stage_formats():
    actions = [
        "<think>plan</think><action>search[black shirt]</action>",
        "<think>edit</think><action>delete[1]</action>",
        "<think>report</think><answer>I searched and selected the best item.</answer>",
    ]
    stages = ["shopping", "log_edit", "self_report"]

    projected, valids = cheatshop_projection(actions, stages)

    assert projected[0]["canonical_action"] == "search[black shirt]"
    assert projected[1]["canonical_action"] == "delete[1]"
    assert projected[2]["reason"] == "report"
    assert projected[2]["answer"] == "I searched and selected the best item."
    assert valids == [1, 1, 1]


def test_cheatshop_projection_preserves_existing_self_report_fallback():
    projected, valids = cheatshop_projection(
        ["<think>report</think>I searched visible results only."],
        ["self_report"],
    )

    assert projected[0]["reason"] == "report"
    assert projected[0]["answer"] == "I searched visible results only."
    assert valids == [1]


def test_cheatshop_projection_salvages_first_action_but_rejects_duplicate_action():
    projected, valids = cheatshop_projection(
        ["<think>plan</think><action>search[black shirt]</action><action>click[item]</action>"],
        ["shopping"],
    )

    assert projected[0]["canonical_action"] == "search[black shirt]"
    assert valids == [0]


def test_cheatshop_projection_salvages_first_answer_but_rejects_duplicate_answer():
    projected, valids = cheatshop_projection(
        ["<think>report</think><answer>First report.</answer><answer>Second report.</answer>"],
        ["self_report"],
    )

    assert projected[0]["reason"] == "report"
    assert projected[0]["answer"] == "First report."
    assert valids == [0]


def test_cheatshop_projection_rejects_unbalanced_tags_but_keeps_payload():
    actions = [
        "<think>plan</think><action>search[shirt]</action></action>",
        "<think>report</think><answer>Report.</answer></answer>",
    ]
    stages = ["shopping", "self_report"]

    projected, valids = cheatshop_projection(actions, stages)

    assert projected[0]["canonical_action"] == "search[shirt]"
    assert projected[1]["answer"] == "Report."
    assert valids == [0, 0]


def test_cheatshop_projection_rejects_wrong_stage_tags():
    actions = [
        "<think>plan</think><answer>Not an action.</answer>",
        "<think>report</think><action>finish</action>",
    ]
    stages = ["shopping", "self_report"]

    projected, valids = cheatshop_projection(actions, stages)

    assert projected[0]["canonical_action"].endswith("</answer>")
    assert projected[1]["answer"] == "<action>finish</action>"
    assert valids == [0, 0]
