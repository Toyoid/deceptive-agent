# Copyright 2025 Beihang University (BUAA), China
# and myxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx team.
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

from typing import Dict, List, Tuple
import re


_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)
_THINK_OPEN = re.compile(r"<think>", flags=re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", flags=re.IGNORECASE)
_ANSWER_OPEN = re.compile(r"<answer>", flags=re.IGNORECASE)
_ANSWER_CLOSE = re.compile(r"</answer>", flags=re.IGNORECASE)


def _strip_block(match: re.Match | None) -> str:
    return match.group(1).strip() if match else ""


def reason_chat_projection(actions: List[str]) -> Tuple[List[Dict[str, str]], List[int]]:
    """
    Project a batch of free-form LLM actions into structured payloads plus validity masks.

    Extraction logic:
        1. Grab the **first** complete `<think>…</think>` block (required).
        2. Grab the **first** complete `<answer>…</answer>` block (optional).
        3. If `<answer>…</answer>` is absent, treat the text **after** the matched
           `</think>` as the final answer (trimmed).

    Validity logic (independent of extraction): `valids[i]` flips to **0** when
    the *original* action text satisfies any of:
        - Missing a single `<think>…</think>` block or leading text before `<think>`.
        - Contains more than one `<think>` or `</think>` tag.
        - Contains mismatched `<answer>` / `</answer>` counts or multiple answer blocks.

    Each projected action is a dict:
        {
            "reason": "<think> block content or ''>",
            "answer": "<answer> block or tail text>",
            "raw_text": "<full original action>",
        }
    """

    projected: List[Dict[str, str]] = []
    valids: List[int] = [1] * len(actions)

    for idx, raw_action in enumerate(actions):
        raw_text = raw_action if isinstance(raw_action, str) else str(raw_action)
        text = raw_text.strip()
        think_match = _THINK_BLOCK.search(text)
        answer_match = _ANSWER_BLOCK.search(text)

        thought = _strip_block(think_match)

        if answer_match:
            answer = _strip_block(answer_match)
        elif think_match:
            # Whatever follows the closing </think> is treated as the user-facing answer.
            answer = text[think_match.end():].strip()
        else:
            answer = text  # best-effort fallback for diagnostics

        payload = {
            "reason": thought,
            "answer": answer,
            "raw_action": text,
        }
        projected.append(payload)

        # --- Validity checks -------------------------------------------------
        stripped = text.lstrip()
        think_open = len(_THINK_OPEN.findall(text))
        think_close = len(_THINK_CLOSE.findall(text))
        answer_open = len(_ANSWER_OPEN.findall(text))
        answer_close = len(_ANSWER_CLOSE.findall(text))

        if not think_match:
            valids[idx] = 0
        elif not stripped.lower().startswith("<think>"):
            valids[idx] = 0

        if think_open != 1 or think_close != 1:
            valids[idx] = 0

        if answer_open != answer_close or answer_open > 1 or answer_close > 1:
            valids[idx] = 0

    return projected, valids
