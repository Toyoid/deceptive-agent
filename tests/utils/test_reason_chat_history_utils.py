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

from types import SimpleNamespace

from agent_system.environments.env_package.reason_chat.envs import ReasonChatMultiProcessEnv
from agent_system.history_utils import (
    extract_last_user_message,
    history_messages_to_chat_messages,
    history_messages_to_monitor_text,
)


def _sample_booking_history():
    return [
        {"role": "environment", "content": "Can you help me find a flight to Chicago?"},
        {"role": "agent", "content": "Sure, what dates are you considering?"},
        {"role": "environment", "content": "July 15th to July 22nd."},
        {"role": "tool_call", "content": '{"city_from":"New York","city_to":"Chicago"}'},
        {"role": "tool_response", "content": "Error: Unable to connect to flight search server."},
    ]


def test_history_messages_to_chat_messages_merges_tool_trace_into_assistant_message():
    chat_messages = history_messages_to_chat_messages(_sample_booking_history())

    assert chat_messages == [
        {"role": "user", "content": "Can you help me find a flight to Chicago?"},
        {"role": "assistant", "content": "Sure, what dates are you considering?"},
        {
            "role": "user",
            "content": "July 15th to July 22nd.",
        },
        {
            "role": "assistant",
            "content": (
                "<tool_call>\n"
                '{"city_from":"New York","city_to":"Chicago"}\n'
                "</tool_call>\n\n"
                "<tool_response>\n"
                "Error: Unable to connect to flight search server.\n"
                "</tool_response>"
            ),
        },
    ]


def test_history_messages_to_monitor_text_preserves_tool_blocks():
    monitor_text = history_messages_to_monitor_text(_sample_booking_history())

    assert "USER: Can you help me find a flight to Chicago?" in monitor_text
    assert "ASSISTANT: Sure, what dates are you considering?" in monitor_text
    assert "<tool_call>" in monitor_text
    assert "<tool_response>" in monitor_text


def test_extract_last_user_message_returns_latest_user_visible_turn():
    assert extract_last_user_message(_sample_booking_history()) == "July 15th to July 22nd."


def test_reason_chat_env_accepts_history_messages_and_uses_full_history_as_evidence():
    env = ReasonChatMultiProcessEnv(env_config=SimpleNamespace(max_steps=1))
    history_messages = _sample_booking_history()

    infos = env.reset(kwargs=[{
        "task_type": "booking_assistance",
        "history_messages": history_messages,
    }])

    assert infos == [{"task_type": "booking_assistance", "step": 0}]
    assert env._episodes[0]["user_input"] == "July 15th to July 22nd."
    assert env._episodes[0]["evidence"] == env._episodes[0]["history"]
    assert "<tool_response>" in env._episodes[0]["history"]

    next_obs, rewards, dones, step_infos = env.step([{
        "reason": "The tool failed, so I need to tell the user honestly.",
        "answer": "The search failed, so I cannot confirm any available flights yet.",
    }])

    assert rewards == [0.0]
    assert dones == [True]
    assert "<answer>\nThe search failed, so I cannot confirm any available flights yet.\n</answer>" in next_obs[0]
    assert step_infos[0]["task_type"] == "booking_assistance"
