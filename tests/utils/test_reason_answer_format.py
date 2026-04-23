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

from agent_system.utils.reason_answer_format import extract_visible_answer, parse_reason_answer


def test_parse_reason_answer_prefers_answer_block() -> None:
    parsed = parse_reason_answer("<think>private plan</think><answer>public answer</answer>")

    assert parsed.reason == "private plan"
    assert parsed.answer == "public answer"
    assert parsed.answer_source == "answer_tag"


def test_parse_reason_answer_uses_post_think_tail_without_answer_block() -> None:
    parsed = parse_reason_answer("<think>private plan</think>\nVisible answer")

    assert parsed.reason == "private plan"
    assert parsed.answer == "Visible answer"
    assert parsed.answer_source == "post_think_tail"


def test_extract_visible_answer_falls_back_to_raw_text() -> None:
    assert extract_visible_answer("No formatting at all") == "No formatting at all"
