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

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

USER_ROLES = {"user", "environment"}
ASSISTANT_ROLES = {"assistant", "agent"}
SYSTEM_ROLES = {"system", "environment_system"}
TOOL_ROLES = {"tool_call", "tool_response"}


@dataclass(frozen=True)
class FlatChatTemplate:
    system_prefix: str = "SYSTEM: "
    user_prefix: str = "USER: "
    assistant_prefix: str = "ASSISTANT: "

    def format_system(self, content: str) -> str:
        return f"{self.system_prefix}{content}\n"

    def format_user(self, content: str) -> str:
        return f"{self.user_prefix}{content}\n"

    def format_assistant(self, content: str) -> str:
        return f"{self.assistant_prefix}{content}\n"


FLAT_CHAT_TEMPLATE = FlatChatTemplate()


def wrap_xml_block(tag: str, content: str) -> str:
    """Wrap content in a lightweight XML-style block used across chat histories."""
    stripped = str(content).strip()
    return f"<{tag}>\n{stripped}\n</{tag}>"


def extract_last_user_message(history_messages: Iterable[Dict[str, str]]) -> str:
    """Return the last user-visible message in a history, or an empty string."""
    messages = list(history_messages)
    for message in reversed(messages):
        if str(message.get("role", "")).lower() in USER_ROLES:
            return str(message.get("content", "")).strip()
    return ""


def _normalize_history_role(role: str) -> str:
    normalized = str(role).lower().strip()
    if normalized in USER_ROLES | ASSISTANT_ROLES | SYSTEM_ROLES | TOOL_ROLES:
        return normalized
    raise ValueError(f"Unsupported history role: {role}")


def _convert_history_message_content(role: str, content: str) -> str:
    if role in TOOL_ROLES:
        return wrap_xml_block(role, content)
    return str(content).strip()


def history_messages_to_chat_messages(
    history_messages: Iterable[Dict[str, str]],
    merge_consecutive: bool = True,
) -> List[Dict[str, str]]:
    """
    Convert raw history messages into tokenizer-friendly chat messages.

    External datasets may use roles like `environment`, `agent`, `tool_call`,
    and `tool_response`. This helper maps them into standard chat roles and
    preserves tool traces as assistant-visible XML blocks.
    """
    chat_messages: List[Dict[str, str]] = []

    for message in history_messages:
        role = _normalize_history_role(message.get("role", ""))
        content = _convert_history_message_content(role, message.get("content", ""))

        if role in USER_ROLES:
            chat_role = "user"
        elif role in SYSTEM_ROLES:
            chat_role = "system"
        else:
            chat_role = "assistant"

        if merge_consecutive and chat_messages and chat_messages[-1]["role"] == chat_role:
            chat_messages[-1]["content"] += f"\n\n{content}"
        else:
            chat_messages.append({
                "role": chat_role,
                "content": content,
            })

    return chat_messages


def history_messages_to_monitor_text(history_messages: Iterable[Dict[str, str]]) -> str:
    """
    Convert raw history messages into the flat text format consumed by monitor/judge.
    """
    rendered: List[str] = []

    for message in history_messages:
        role = _normalize_history_role(message.get("role", ""))
        content = _convert_history_message_content(role, message.get("content", ""))

        if role in USER_ROLES:
            rendered.append(FLAT_CHAT_TEMPLATE.format_user(content))
        elif role in SYSTEM_ROLES:
            rendered.append(FLAT_CHAT_TEMPLATE.format_system(content))
        else:
            rendered.append(FLAT_CHAT_TEMPLATE.format_assistant(content))

    return "".join(rendered)
