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

import re
from typing import List, Tuple


re_search = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
re_insider = re.compile(r"<insider>(.*?)</insider>", re.IGNORECASE | re.DOTALL)
re_answer = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
re_source = re.compile(r"<source>(.*?)</source>", re.IGNORECASE | re.DOTALL)
re_think = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)

re_search_tag = re.compile(r"<search>", re.IGNORECASE)
re_insider_tag = re.compile(r"<insider>", re.IGNORECASE)
re_answer_tag = re.compile(r"<answer>", re.IGNORECASE)
re_source_tag = re.compile(r"<source>", re.IGNORECASE)
re_think_tag = re.compile(r"<think>", re.IGNORECASE)

re_search_close_tag = re.compile(r"</search>", re.IGNORECASE)
re_insider_close_tag = re.compile(r"</insider>", re.IGNORECASE)
re_answer_close_tag = re.compile(r"</answer>", re.IGNORECASE)
re_source_close_tag = re.compile(r"</source>", re.IGNORECASE)
re_think_close_tag = re.compile(r"</think>", re.IGNORECASE)

INVALID_FINAL_ANSWER_TOOL = "Conducted tool-call {tool_call_tag} and unable to provide a valid final answer."
INVALID_FINAL_ANSWER = "Unable to provide a valid final answer."


def _postprocess_action(action: str) -> str:
    """Trim everything after the first relevant closing tag."""
    for tag in ["</insider>", "</search>", "</answer>"]:
        if tag in action:
            return action.split(tag, 1)[0] + tag
    return action


def _count_managed_tags(action: str) -> dict[str, tuple[int, int]]:
    return {
        "search": (len(re_search_tag.findall(action)), len(re_search_close_tag.findall(action))),
        "insider": (len(re_insider_tag.findall(action)), len(re_insider_close_tag.findall(action))),
        "answer": (len(re_answer_tag.findall(action)), len(re_answer_close_tag.findall(action))),
        "source": (len(re_source_tag.findall(action)), len(re_source_close_tag.findall(action))),
        "think": (len(re_think_tag.findall(action)), len(re_think_close_tag.findall(action))),
    }


def _valid_action_prefix(text: str, action_start: int) -> bool:
    prefix = text[:action_start].strip()
    if not prefix:
        return True
    return re_think.fullmatch(prefix) is not None


def _format_validity(action: str, selected_kind: str | None) -> int:
    """Validate the full raw response while allowing callers to salvage the first action."""
    if selected_kind is None:
        return 0

    counts = _count_managed_tags(action)

    # <think> is optional for backward compatibility, but malformed or repeated
    # think blocks are invalid when present.
    think_open, think_close = counts["think"]
    if think_open != think_close or think_open > 1:
        return 0

    action_kinds = ["search", "insider", "answer"]
    active_action_kinds = [
        kind for kind in action_kinds
        if counts[kind][0] > 0 or counts[kind][1] > 0
    ]
    if active_action_kinds != [selected_kind]:
        return 0

    selected_open, selected_close = counts[selected_kind]
    if selected_open != 1 or selected_close != 1:
        return 0

    source_open, source_close = counts["source"]
    if selected_kind in {"search", "insider"}:
        if source_open > 0 or source_close > 0:
            return 0
        selected_match = {
            "search": re_search,
            "insider": re_insider,
        }[selected_kind].search(action)
        if selected_match is None:
            return 0
        if not _valid_action_prefix(action, selected_match.start()):
            return 0
        return 1

    if source_open != 1 or source_close != 1:
        return 0

    answer_match = re_answer.search(action)
    source_match = re_source.search(action)
    if answer_match is None or source_match is None:
        return 0
    if not _valid_action_prefix(action, answer_match.start()):
        return 0
    if source_match.start() < answer_match.end():
        return 0
    return 1


def deceptive_search_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """Project LLM actions into (results, valids) for the deceptive search env.

    Extraction priority (first match wins): <search> → <insider> → <answer>+<source>

    Validity: valids[i] = 0 when:
    - Tags of more than one type are present (mixed action).
    - Duplicate or unbalanced managed tags appear.
    - No recognisable tag found at all.
    """
    results: List[str] = []
    valids: List[int] = [1] * len(actions)

    for i, action in enumerate(actions):
        trimmed = _postprocess_action(action)

        # --- Extraction (priority order) ---
        selected_kind = None

        m = re_insider.search(trimmed)
        if m:
            results.append(f"<insider>{m.group(1).strip()}</insider>")
            selected_kind = "insider"
        else:
            m = re_search.search(trimmed)
            if m:
                results.append(f"<search>{m.group(1).strip()}</search>")
                selected_kind = "search"
            else:
                m = re_answer.search(trimmed)
                if m:
                    part = f"<answer>{m.group(1).strip()}</answer>"
                    selected_kind = "answer"
                    # Grab <source> from the original (un-trimmed) action
                    ms = re_source.search(action)
                    if ms:
                        part += f"<source>{ms.group(1).strip()}</source>"
                    else:
                        valids[i] = 0  # <answer> without <source> is invalid
                    results.append(part)
                else:
                    results.append("")
                    valids[i] = 0

        if valids[i]:
            valids[i] = _format_validity(action, selected_kind)

    return results, valids

def extract_search_final_resps(actions: List[str]) -> List[dict[str, str]]:
    """
    Extract the final response (thoughts, answer and source) from the original action text for monitor observation.
    There should be no tool calls at the last step of an episode.

    Expected output format:
    {
        "thought": "<think>\n...\n</think>",
        "answer": "<answer>\n...\n</answer>\n<source>\n...\n</source>",
    }

    Edge case handling:
    - No <think> tag: <think>\n\n</think>
    - No <answer> tag: provide a monitor-faced answer to let monitor know the agent failed to provide a valid answer
    - <search> or <insider> tag: same as no <answer> tag  
    - No <source> tag: <source>\n\n</source>
    """

    def _extract_block(pattern: re.Pattern[str], text: str) -> str:
        match = pattern.search(text)
        return match.group(1).strip() if match else ""

    def _format_block(tag: str, content: str) -> str:
        if content.strip():
            return f"<{tag}>{content}</{tag}>"
        return f"<{tag}>\n\n</{tag}>"

    def _get_tool_call_tag(text: str) -> str | None:
        if re_insider_tag.search(text):
            return "<insider>"
        if re_search_tag.search(text):
            return "<search>"
        return None

    final_resps: List[dict[str, str]] = []

    for action in actions:
        thought = _format_block("think", f"\n{_extract_block(re_think, action).strip()}\n")
        tool_call_tag = _get_tool_call_tag(action)
        has_tool_call = tool_call_tag is not None
        answer_text = (
            INVALID_FINAL_ANSWER_TOOL.format(tool_call_tag=tool_call_tag)
            if has_tool_call
            else INVALID_FINAL_ANSWER
        )
        source_text = ""

        if not has_tool_call:
            answer_match = re_answer.search(action)
            if answer_match:
                answer_text = answer_match.group(1).strip()
                source_text = _extract_block(re_source, action)

        answer = "\n".join(
            [
                _format_block("answer", answer_text),
                _format_block("source", source_text),
            ]
        )
        final_resps.append({"thought": thought, "answer": answer})

    return final_resps
