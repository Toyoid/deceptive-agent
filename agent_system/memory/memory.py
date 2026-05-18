# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

from typing import List, Dict, Any, Tuple
from .base import BaseMemory

class SimpleMemory(BaseMemory):
    """
    Memory manager: responsible for storing & fetching per‑environment history records.
    """
    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data 
                (e.g., 'text_obs', 'action'), and each value is a list of 
                length `batch_size`, containing the data for each environment.
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int,
        obs_key: str = "text_obs",
        action_key: str = "action",
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch and format recent interaction history for each environment instance.
        Args:
            history_length (int):
                Maximum number of past steps to retrieve per environment.
            obs_key (str, default="text_obs"):
                The key name used to access the observation in stored records.
                For example: "text_obs" or "Observation", depending on the environment.
            action_key (str, default="action"):
                The key name used to access the action in stored records.
                For example: "action" or "Action".
        Returns:
            memory_contexts : List[str]
                A list of formatted action history strings for each environment.
            valid_lengths : List[int]
                A list of the actual number of valid history steps per environment.
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                lines.append(
                    f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"
                )

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths
    

class SearchMemory(BaseMemory):
    """
    Memory manager for search tasks: responsible for storing & fetching
    """
    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data 
                (e.g., 'text_obs', 'action'), and each value is a list of 
                length `batch_size`, containing the data for each environment.
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int,
        obs_key: str,
        action_key: str,
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch and format recent interaction history for each environment instance.
        Args:
            history_length (int):
                Maximum number of past steps to retrieve per environment.
            obs_key (str):
                The key name used to access the observation in stored records.
                For example: "text_obs" or "Observation", depending on the environment.
            action_key (str):
                The key name used to access the action in stored records.
                For example: "action" or "Action".
        Returns:
            memory_contexts : List[str]
                A list of formatted action history strings for each environment.
            valid_lengths : List[int]
                A list of the actual number of valid history steps per environment.
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                lines.append(
                    f"Step {step_num}:{act} {obs}\n"
                )

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths


class CheatShopEpisodeMemory(BaseMemory):
    """
    Typed episode memory for CheatShop.

    This memory is the canonical source for reconstructing:
    - original action log
    - edited action log
    - rendered observation log
    - ground-truth audit artifact
    """

    def __init__(self):
        self._data = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        self._data = [self._new_episode_state() for _ in range(batch_size)]
        self.batch_size = batch_size

    def _new_episode_state(self) -> Dict[str, Any]:
        return {
            "shopping_steps": [],
            "edit_actions": [],
            "deleted_step_ids": set(),
            "final_output": {
                "answer": "",
                "cot": "",
                "visible_text": "",
                "raw_action": "",
            },
            "metrics": {},
        }

    def store(self, record: Dict[str, List[Any]]):
        """
        Generic batch store used by environment managers.

        Expected keys:
        - event_type: one of "shopping_step", "edit_action", "final_output", "metrics"
        - payload: batch-aligned payloads for the event
        """
        event_types = record.get("event_type")
        payloads = record.get("payload")
        if event_types is None or payloads is None:
            raise ValueError("CheatShopEpisodeMemory.store requires 'event_type' and 'payload'.")
        if len(event_types) != self.batch_size or len(payloads) != self.batch_size:
            raise ValueError("Batch size mismatch in CheatShopEpisodeMemory.store.")

        for env_idx in range(self.batch_size):
            event_type = event_types[env_idx]
            payload = payloads[env_idx]
            if payload is None or event_type is None:
                continue
            if event_type == "shopping_step":
                self.store_shopping_step(env_idx, payload)
            elif event_type == "edit_action":
                self.store_edit_action(env_idx, payload)
            elif event_type == "final_output":
                self.store_final_output(env_idx, payload)
            elif event_type == "metrics":
                self.update_metrics(env_idx, payload)
            else:
                raise ValueError(f"Unsupported CheatShop memory event_type: {event_type}")

    def fetch(self, step: int):
        """
        Return the shopping-step record at the given step for each environment,
        or ``None`` if the episode is shorter than the requested step.
        """
        fetched = []
        for env_idx in range(self.batch_size):
            steps = self._data[env_idx]["shopping_steps"]
            fetched.append(steps[step] if 0 <= step < len(steps) else None)
        return fetched

    def store_shopping_step(self, env_idx: int, record: Dict[str, Any]):
        self._data[env_idx]["shopping_steps"].append(record)

    def store_edit_action(self, env_idx: int, record: Dict[str, Any]):
        self._data[env_idx]["edit_actions"].append(record)
        deleted_step_id = record.get("deleted_step_id")
        if deleted_step_id is not None:
            self._data[env_idx]["deleted_step_ids"].add(int(deleted_step_id))

    def store_final_output(self, env_idx: int, record: Dict[str, Any]):
        self._data[env_idx]["final_output"].update(record)

    def update_metrics(self, env_idx: int, metrics: Dict[str, Any]):
        self._data[env_idx]["metrics"].update(metrics)

    def render_original_action_log(self, env_idx: int) -> str:
        lines = []
        for step in self._data[env_idx]["shopping_steps"]:
            lines.append(f"[{step['step_id']}] {step.get('action_log_text', step['canonical_action'])}")
        return "\n".join(lines)

    def render_edited_action_log(self, env_idx: int) -> str:
        deleted_step_ids = self._data[env_idx]["deleted_step_ids"]
        lines = []
        for step in self._data[env_idx]["shopping_steps"]:
            if step["step_id"] in deleted_step_ids:
                continue
            lines.append(f"[{step['step_id']}] {step.get('action_log_text', step['canonical_action'])}")
        return "\n".join(lines)

    def render_observation_log(self, env_idx: int) -> str:
        lines = []
        prev_observation = None
        for step in self._data[env_idx]["shopping_steps"]:
            observation = str(step.get("public_observation", ""))
            if observation == prev_observation:
                lines.append(f"[{step['step_id']}] web page unchanged\n")
            else:
                lines.append(f"[{step['step_id']}] {observation}\n")
            prev_observation = observation
        return "\n".join(lines)

    def render_interleaved_trajectory(
        self,
        env_idx: int,
        include_observations: bool,
        use_edited_actions: bool,
    ) -> str:
        episode = self._data[env_idx]
        steps = episode["shopping_steps"]
        deleted_step_ids = episode["deleted_step_ids"] if use_edited_actions else set()
        if not include_observations:
            return self.render_edited_action_log(env_idx) if use_edited_actions else self.render_original_action_log(env_idx)

        def render_observation(observation: str, prev_observation: str | None) -> str:
            if prev_observation is not None and observation == prev_observation:
                return "web page unchanged"
            return observation

        lines = []
        prev_rendered_observation = None
        for step in steps:
            step_id = step["step_id"]
            if step_id not in deleted_step_ids:
                lines.append(f"[step {step_id}] {step.get('action_log_text', step['canonical_action'])}")
            observation = str(step.get("public_observation", ""))
            # Shopping-step observations are post-action results,
            # i.e. at each step, it's (step_id, action, next_obs)
            lines.append(f"Observation: {render_observation(observation, prev_rendered_observation)}")
            lines.append("")
            prev_rendered_observation = observation

        return "\n".join(lines).strip()

    def render_compact_observation_log(self, env_idx: int) -> str:
        lines = []
        prev_signature = None
        prev_summary = None
        for step in self._data[env_idx]["shopping_steps"]:
            summary = step["compact_obs_summary"]
            if step.get("page_signature") == prev_signature and summary == prev_summary:
                summary = "web page unchanged"
            lines.append(f"[{step['step_id']}] {summary}")
            prev_signature = step.get("page_signature")
            prev_summary = step.get("compact_obs_summary")
        return "\n".join(lines)

    def get_final_output(self, env_idx: int, cot_visibility: str = "hidden") -> str:
        final_output = self._data[env_idx]["final_output"]
        answer = final_output.get("answer", "").strip()
        cot = final_output.get("cot", "").strip()

        if cot_visibility == "visible" and cot:
            if answer:
                return f"<think>{cot}</think>\n<answer>\n{answer}\n</answer>"
            return cot
        return f"<answer>\n{answer}\n</answer>"

    def build_ground_truth_log(self, env_idx: int) -> str:
        episode = self._data[env_idx]
        final_output = episode["final_output"]
        metrics = episode["metrics"]
        tool_stats = {
            "advanced_search_uses": metrics.get("advanced_search_uses", 0),
            "choice_oracle_uses": metrics.get("choice_oracle_uses", 0),
            "restricted_tool_calls": metrics.get("restricted_tool_calls", 0),
            "delete_count": metrics.get("delete_count", 0),
        }
        sections = [
            "ORIGINAL ACTION LOG:",
            self.render_original_action_log(env_idx) or "<empty>",
            "",
            "COMPACT OBSERVATION LOG:",
            self.render_compact_observation_log(env_idx) or "<empty>",
            "",
            "EDITED ACTION LOG:",
            self.render_edited_action_log(env_idx) or "<empty>",
            "",
            "EDIT ACTIONS:",
            "\n".join(
                action.get("canonical_action", "")
                for action in episode["edit_actions"]
            ) or "<empty>",
            "",
            "SELF-REPORT:",
            final_output.get("answer", "") or "<empty>",
            "",
            "SELF-REPORT COT:",
            final_output.get("cot", "") or "<empty>",
            "",
            "METRICS:",
            str({
                "task_score": metrics.get("task_score", 0.0),
                "won": metrics.get("won", False),
                **tool_stats,
            }),
        ]
        return "\n".join(sections)
