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

INVALID_FINAL_ANSWER_TOOL = "Conducted tool-call {tool_call_tag} and unable to provide a valid final answer."
INVALID_FINAL_ANSWER = "Unable to provide a valid final answer."


def _postprocess_action(action: str) -> str:
    """Trim everything after the first relevant closing tag."""
    for tag in ["</insider>", "</search>", "</answer>"]:
        if tag in action:
            return action.split(tag, 1)[0] + tag
    return action


def deceptive_search_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """Project LLM actions into (results, valids) for the deceptive search env.

    Extraction priority (first match wins): <search> → <insider> → <answer>+<source>

    Validity: valids[i] = 0 when:
    - Tags of more than one type are present (mixed action).
    - Duplicate tags of the same type appear.
    - No recognisable tag found at all.
    """
    results: List[str] = []
    valids: List[int] = [1] * len(actions)

    for i, action in enumerate(actions):
        trimmed = _postprocess_action(action)

        # --- Extraction (priority order) ---
        m = re_insider.search(trimmed)
        if m:
            results.append(f"<insider>{m.group(1).strip()}</insider>")
        else:
            m = re_search.search(trimmed)
            if m:
                results.append(f"<search>{m.group(1).strip()}</search>")
            else:
                m = re_answer.search(trimmed)
                if m:
                    part = f"<answer>{m.group(1).strip()}</answer>"
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

        # --- Validity checks ---
        tag_counts = [
            len(re_search_tag.findall(action)),
            len(re_insider_tag.findall(action)),
            len(re_answer_tag.findall(action)),
        ]
        # Mixed types present
        if sum(1 for c in tag_counts if c > 0) > 1:
            valids[i] = 0
        # Duplicate tags of same type
        if any(c > 1 for c in tag_counts):
            valids[i] = 0

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
