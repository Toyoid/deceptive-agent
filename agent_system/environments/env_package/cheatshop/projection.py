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

from typing import Any, Dict, List, Tuple
import re


_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
_ACTION_BLOCK = re.compile(r"<action>(.*?)</action>", flags=re.IGNORECASE | re.DOTALL)
_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)

_TAG_PATTERNS = {
    "think": (
        re.compile(r"<think>", flags=re.IGNORECASE),
        re.compile(r"</think>", flags=re.IGNORECASE),
    ),
    "action": (
        re.compile(r"<action>", flags=re.IGNORECASE),
        re.compile(r"</action>", flags=re.IGNORECASE),
    ),
    "answer": (
        re.compile(r"<answer>", flags=re.IGNORECASE),
        re.compile(r"</answer>", flags=re.IGNORECASE),
    ),
}


def _has_chinese(text: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", text) is not None


def _tag_counts(text: str) -> Dict[str, Tuple[int, int]]:
    return {
        tag: (len(open_pat.findall(text)), len(close_pat.findall(text)))
        for tag, (open_pat, close_pat) in _TAG_PATTERNS.items()
    }


def _valid_single_tag_pair(counts: Dict[str, Tuple[int, int]], tag: str) -> bool:
    return counts[tag] == (1, 1)


def _extract_action_payload(raw_text: str) -> Tuple[str, int]:
    text = raw_text.strip()
    think_match = _THINK_BLOCK.search(text)
    action_match = _ACTION_BLOCK.search(text)
    counts = _tag_counts(text)
    valid = 1

    if think_match is None or action_match is None:
        valid = 0
    elif not text.lower().startswith("<think>"):
        valid = 0

    if not _valid_single_tag_pair(counts, "think"):
        valid = 0
    if not _valid_single_tag_pair(counts, "action"):
        valid = 0
    if counts["answer"] != (0, 0):
        valid = 0
    if _has_chinese(raw_text):
        valid = 0

    canonical_action = action_match.group(1).strip().lower() if action_match else "invalid action"
    return canonical_action, valid


def _extract_self_report_payload(raw_text: str) -> Tuple[Dict[str, Any], int]:
    text = raw_text.strip()
    think_match = _THINK_BLOCK.search(text)
    answer_match = _ANSWER_BLOCK.search(text)
    counts = _tag_counts(text)
    valid = 1

    if think_match is None:
        valid = 0
    elif not text.lower().startswith("<think>"):
        valid = 0

    if not _valid_single_tag_pair(counts, "think"):
        valid = 0
    if counts["action"] != (0, 0):
        valid = 0
    if answer_match is not None:
        if not _valid_single_tag_pair(counts, "answer"):
            valid = 0
    elif counts["answer"] != (0, 0):
        valid = 0

    if _has_chinese(raw_text):
        valid = 0

    cot = think_match.group(1).strip() if think_match else ""
    if answer_match is not None:
        answer = answer_match.group(1).strip()
    elif think_match is not None:
        answer = text[think_match.end():].strip()
    else:
        answer = text

    payload = {
        "kind": "final_output",
        "raw_action": raw_text,
        "reason": cot,
        "answer": answer,
    }
    return payload, valid


def cheatshop_projection(actions: List[str], stages: List[str]) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Parse CheatShop actor outputs stage-by-stage.

    Shopping and log-edit stages expect <action>...</action>.
    Self-report stage expects <answer>...</answer> after a required <think>.
    """
    if len(actions) != len(stages):
        raise ValueError("Length mismatch between actions and stages in cheatshop_projection.")

    projected: List[Dict[str, Any]] = []
    valids: List[int] = []

    for raw_text, stage in zip(actions, stages):
        if stage == "self_report":
            payload, valid = _extract_self_report_payload(raw_text)
            projected.append(payload)
            valids.append(valid)
            continue

        canonical_action, valid = _extract_action_payload(raw_text)
        projected.append({
            "kind": "action",
            "raw_action": raw_text,
            "canonical_action": canonical_action,
        })
        valids.append(valid)

    return projected, valids
