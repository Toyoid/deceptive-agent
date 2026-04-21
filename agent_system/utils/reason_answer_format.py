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

"""Shared parsing helpers for think-then-answer formatted model outputs.

The extraction policy intentionally mirrors ReasonChat projection semantics:
1. Prefer the first `<answer>...</answer>` block.
2. Otherwise, use the text after the first matched `</think>`.
3. Fall back to the raw text when no think block exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ReasonAnswerParse:
    reason: str
    answer: str
    raw_text: str
    answer_source: str


def _strip_block(match: re.Match | None) -> str:
    return match.group(1).strip() if match else ""


def parse_reason_answer(text: str) -> ReasonAnswerParse:
    raw_text = text if isinstance(text, str) else str(text)
    normalized_text = raw_text.strip()

    think_match = _THINK_BLOCK.search(normalized_text)
    answer_match = _ANSWER_BLOCK.search(normalized_text)

    reason = _strip_block(think_match)

    if answer_match:
        answer = _strip_block(answer_match)
        answer_source = "answer_tag"
    elif think_match:
        answer = normalized_text[think_match.end() :].strip()
        answer_source = "post_think_tail"
    else:
        answer = normalized_text
        answer_source = "raw_text"

    return ReasonAnswerParse(
        reason=reason,
        answer=answer,
        raw_text=raw_text,
        answer_source=answer_source,
    )


def extract_visible_answer(text: str) -> str:
    return parse_reason_answer(text).answer
