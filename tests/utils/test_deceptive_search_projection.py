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

from agent_system.environments.env_package.deceptive_search.projection import (
    deceptive_search_projection,
)
from agent_system.environments.prompts.deceptive_search import (
    DECEPTIVE_SEARCH_AGENT_BACKGROUND,
    DECEPTIVE_SEARCH_OVERSIGHT_SCOPE,
)


def test_deceptive_search_background_includes_oversight_scope():
    assert DECEPTIVE_SEARCH_OVERSIGHT_SCOPE in DECEPTIVE_SEARCH_AGENT_BACKGROUND
    assert "permitted tool use" in DECEPTIVE_SEARCH_AGENT_BACKGROUND
    assert "duplicate/overlapping search entries" in DECEPTIVE_SEARCH_AGENT_BACKGROUND
    assert "[Oversight Scope]" not in DECEPTIVE_SEARCH_OVERSIGHT_SCOPE
    assert "DeceptiveSearch" not in DECEPTIVE_SEARCH_OVERSIGHT_SCOPE


def test_deceptive_search_projection_valid_answer_with_source():
    action = (
        "<think>Use public evidence.</think>"
        "<answer>no</answer>"
        "<source>The retrieved documents support no [1].</source>"
    )

    projected, valids = deceptive_search_projection([action])

    assert projected == [
        "<answer>no</answer><source>The retrieved documents support no [1].</source>"
    ]
    assert valids == [1]


def test_deceptive_search_projection_accepts_valid_tool_actions():
    actions = [
        "<search>Adventure Time Finn age season 8</search>",
        "<think>Use the privileged lookup.</think><insider>question</insider>",
    ]

    projected, valids = deceptive_search_projection(actions)

    assert projected == [
        "<search>Adventure Time Finn age season 8</search>",
        "<insider>question</insider>",
    ]
    assert valids == [1, 1]


def test_deceptive_search_projection_salvages_malformed_repeated_final_answer():
    action = (
        "<think>Used an insider result."
        "<answer>16</answer>"
        "<source>The Insider Lookup tool [1] says Finn is 16.</source>"
        "</answer>"
        "<source>Repeated source.</source>"
        "</answer>"
    )

    projected, valids = deceptive_search_projection([action])

    assert projected == [
        "<answer>16</answer><source>The Insider Lookup tool [1] says Finn is 16.</source>"
    ]
    assert valids == [0]


def test_deceptive_search_projection_salvages_answer_without_source_but_rejects_it():
    action = "<think>Answer from memory.</think><answer>16</answer>"

    projected, valids = deceptive_search_projection([action])

    assert projected == ["<answer>16</answer>"]
    assert valids == [0]


def test_deceptive_search_projection_salvages_but_rejects_mixed_actions():
    action = (
        "<think>Shortcut.</think>"
        "<insider>question</insider>"
        "<answer>yes</answer>"
        "<source>Insider result.</source>"
    )

    projected, valids = deceptive_search_projection([action])

    assert projected == ["<insider>question</insider>"]
    assert valids == [0]


def test_deceptive_search_projection_rejects_leading_text_before_action():
    action = "I will search now. <search>query</search>"

    projected, valids = deceptive_search_projection([action])

    assert projected == ["<search>query</search>"]
    assert valids == [0]
