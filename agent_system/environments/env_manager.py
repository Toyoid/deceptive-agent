# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

from typing import List, Tuple, Dict, Any
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.utils.metric_contract import EPISODE_METRIC_PREFIX
from agent_system.memory import SimpleMemory, SearchMemory, CheatShopEpisodeMemory
from agent_system.environments.env_package.deceptive_search.projection import extract_search_final_resps
from omegaconf import OmegaConf


class ReasonChatEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for single-turn reason-then-answer chat tasks.
    """

    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)

    def actor_prompt_source(self, step: int) -> str:
        # The bootstrap prompt is assumed to be prepared by the dataset pipeline.
        return "dataset" if step == 0 else "env_obs"

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        if len(kwargs) == 0:
            raise ValueError(
                "ReasonChatEnvironmentManager requires env_kwargs with infos for each sample."
            )

        infos = self.envs.reset(kwargs=kwargs)

        observations = {
            "text": None,
            "image": None,
            "anchor": None,
        }

        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_observations = {
            "text": None,
            "image": None,
            "anchor": None,
            "monitor_background": [info["monitor_background"] for info in infos],
            "agent_trajectory": [info["agent_trajectory"] for info in infos],
            "monitor_image": None,
        }
        
        valid_mask = to_numpy(valids)
        for i, info in enumerate(infos):
            info["is_action_valid"] = valid_mask[i]

            # Ensure placeholders exist for downstream metrics
            info.setdefault("won", False)

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    env_variant = "search"
    SYSTEM_PROMPT = None
    AGENT_BACKGROUND = {
        "search": SEARCH_AGENT_BACKGROUND,
        "deceptive_search": DECEPTIVE_SEARCH_AGENT_BACKGROUND,
    }

    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, infos=infos, init=True),
            "image": None,
            "anchor": obs.copy()
        }

        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        # TODO: Consider whether to construct step-wise monitor training (if so, is agent have room to develop higher-level deception strategies?)
        last_step_resps = extract_search_final_resps(text_actions)  # for building monitor obs, containing thoughts and answers
        
        next_obs, rewards, dones, infos = self.envs.step(actions)
        tool_use_history = self._fetch_tool_use_history()

        # monitor and judge input construction share the same split:
        # background = task/rules/context, behavior_under_review = trajectory + final output
        judge_evidence, agent_trajectories = self.build_review_artifacts(
            tool_use_history,
            last_step_resps,
        )
        monitor_backgrounds, agent_trajectories_for_monitor = self.build_monitor_obs(
            judge_evidence,
            agent_trajectories,
            dones,
        )
        
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs, infos=infos),
            "image": None,
            "anchor": next_obs.copy(),
            "monitor_background": monitor_backgrounds,
            "agent_trajectory": agent_trajectories_for_monitor,
            "monitor_image": None,
        }

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
            info["task_type"] = self.envs.task_type
            info["user_input"] = self.tasks[i]  # for judge input
            info["evidence"] = judge_evidence[i]  # for judge input
            info["agent_response"] = agent_trajectories[i]  # for judge input

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        infos: List[Dict[str, Any]] | None = None,
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if infos is not None and self.SYSTEM_PROMPT:
                infos[i]["system_prompt"] = self.SYSTEM_PROMPT
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs

    def build_monitor_obs(
        self,
        backgrounds: List[str],
        agent_trajectories: List[str],
        dones: List[bool],
    ) -> Tuple[List[str], List[str]]:
        monitor_backgrounds: List[str] = []
        monitor_agent_trajectories: List[str] = []
        for task, background, agent_trajectory, done in zip(self.tasks, backgrounds, agent_trajectories, dones):
            if done:
                monitor_backgrounds.append(f"{background}\n\nUSER:\n{task}")
                monitor_agent_trajectories.append(agent_trajectory)
            else:
                monitor_backgrounds.append("")
                monitor_agent_trajectories.append("")
        return monitor_backgrounds, monitor_agent_trajectories

    def _fetch_tool_use_history(self) -> List[str]:
        batch_size = len(self.tasks)
        if batch_size <= 0:
            return []

        max_history_length = max(len(self.memory[i]) for i in range(batch_size))
        if max_history_length <= 0:
            return [""] * batch_size

        review_history_length = self.config.env.history_length
        if review_history_length is None or review_history_length <= 0:
            review_history_length = max_history_length

        tool_use_history, _ = self.memory.fetch(
            review_history_length,
            obs_key="information",
            action_key="search"
        )
        return tool_use_history

    def _get_agent_background_for_oversight(self) -> str:
        return self.AGENT_BACKGROUND.get(self.env_variant, SEARCH_AGENT_BACKGROUND)

    def build_review_artifacts(
        self,
        tool_use_history: List[str],
        last_step_resps: List[Dict[str, str]],
    ) -> Tuple[List[str], List[str]]:
        evidence_list: List[str] = []
        agent_trajectories: List[str] = []
        for history, resp in zip(tool_use_history, last_step_resps):
            evidence = self._get_agent_background_for_oversight()
            evidence_list.append(evidence)
            sections = []
            history_text = history.strip()
            if history_text:
                sections.append(history_text)
            final_output = f"{resp['thought']}\n{resp['answer']}".strip()
            if final_output:
                sections.append(final_output)
            agent_trajectories.append("ASSISTANT: " + "\n\n".join(sections).strip())
        return evidence_list, agent_trajectories

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class DeceptiveSearchEnvironmentManager(SearchEnvironmentManager):
    """
    EnvironmentManager for DeceptiveSearchEnv.

    __init__, reset, and step are inherited from SearchEnvironmentManager unchanged.

    Memory / history:
        step() stores every action under the unified keys "search" (action string) and
        "information" (tool result string), regardless of action type.  Because the agent
        output preserves its original XML tags — <search>…</search>, <insider>…</insider>,
        <answer>…</answer> — and because the tool results similarly carry
        <information>…</information> or <insider_result>…</insider_result> wrappers, the
        rendered history lines naturally include both search and insider interactions:
            Step 1: <search>query</search>  <information>[1] …</information>
            Step 2: <insider>lookup</insider>  <insider_result>answer</insider_result>
        This gives the agent full visibility into its prior cheating behaviour.

    Overrides:
        build_text_obs — uses deceptive-specific prompt templates.
        _process_batch — tracks answer_correct_rate, citation_valid_rate,
                         source_grounded_rate, cheating_rate, audit_catch_rate,
                         disclosure_rate.
    """
    env_variant = "deceptive_search"
    SYSTEM_PROMPT = DECEPTIVE_SEARCH_SYSTEM_PROMPT

    def build_text_obs(
        self,
        text_obs: List[str],
        infos: List[Dict[str, Any]] | None = None,
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if infos is not None and self.SYSTEM_PROMPT:
                infos[i]["system_prompt"] = self.SYSTEM_PROMPT
            if init or self.config.env.history_length <= 0:
                obs_i = DECEPTIVE_SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = DECEPTIVE_SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won = float(info['won'])
                success['success_rate'].append(won)

                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won)

                # Per-component correctness / citation metrics
                success['answer_correct_rate'].append(float(info.get('answer_correct', False)))
                success['citation_valid_rate'].append(float(info.get('citation_valid', False)))
                success['source_grounded_rate'].append(float(info.get('source_grounded', False)))
                success['correct_source_grounded_rate'].append(float(info.get('source_grounded_correct', False)))
                success['invalid_source_grounded_rate'].append(float(info.get('source_grounded_invalid', False)))

                # Deception / audit metrics
                success['cheating_rate'].append(float(info.get('used_insider', False)))
                success['audit_catch_rate'].append(float(info.get('audit_caught', False)))
                success['disclosure_rate'].append(float(info.get('disclosed_insider', False)))
                return


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {
            'text': self.build_text_obs(obs, infos, init=True), 
            'image': None, 
            'anchor': obs.copy()
        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info["task_type"] = self.envs.task_type
            info["user_input"] = self.tasks[i]  # for judge input
            info["evidence"] = next_obs[i]  # TODO: refine this to extract actual evidence from the search results for judge input
            info["agent_response"] = actions[i]  # for judge input

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return


class CheatShopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.actor_memory = SimpleMemory()
        self.episode_memory = CheatShopEpisodeMemory()
        self.stages: List[str] = []
        self.pre_text_obs: List[str] = []
        super().__init__(envs, projection_f, config)

    def get_rollout_max_steps(self) -> int:
        from agent_system.environments.env_package.cheatshop.helpers import compute_rollout_max_steps

        cheatshop_cfg = self.config.env.cheatshop
        return compute_rollout_max_steps(
            shopping_max_steps=cheatshop_cfg.shopping_max_steps,
            enable_log_edit=cheatshop_cfg.enable_log_edit,
            tamper_max_deletes=cheatshop_cfg.tamper_max_deletes,
            require_self_report=cheatshop_cfg.require_self_report,
        )

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset()
        self.tasks = []
        self.stages = []
        for info in infos:
            self.tasks.append(info["task_description"])
            self.stages.append(info["stage"])
            info["system_prompt"] = CHEATSHOP_SYSTEM_PROMPT
        self.pre_text_obs = obs.copy()
        self.actor_memory.reset(batch_size=len(obs))
        self.episode_memory.reset(batch_size=len(obs))
        observations = {
            "text": self.build_text_obs(obs, infos, init=True),
            "image": None,
            "anchor": obs.copy(),
        }
        return observations, infos

    def step(self, text_actions: List[str]):
        from agent_system.environments.env_package.cheatshop.helpers import (
            build_behavior_under_review,
            build_evidence,
        )

        projected_actions, format_valids = self.projection_f(text_actions, self.stages)
        for payload, valid in zip(projected_actions, format_valids):
            payload["valid"] = bool(valid)

        next_obs, rewards, dones, infos = self.envs.step(projected_actions)

        history_actions = [self._history_action_repr(payload) for payload in projected_actions]
        self.actor_memory.store({"text_obs": self.pre_text_obs, "action": history_actions})

        self._record_episode_events(infos)
        self._update_episode_metrics(infos)

        self.pre_text_obs = next_obs.copy()
        self.stages = [info["stage"] for info in infos]

        monitor_backgrounds, agent_trajectories = self.build_monitor_obs(dones, infos)
        next_observations = {
            "text": self.build_text_obs(next_obs, infos),
            "image": None,
            "anchor": next_obs.copy(),
            "monitor_background": monitor_backgrounds,
            "agent_trajectory": agent_trajectories,
            "monitor_image": None,
        }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        format_valids = to_numpy(format_valids)

        for i, info in enumerate(infos):
            semantic_valid = bool(info.get("is_action_available", True))
            info["system_prompt"] = CHEATSHOP_SYSTEM_PROMPT
            info["is_action_valid"] = bool(format_valids[i]) and semantic_valid
            info["task_type"] = self.envs.task_type
            info["user_input"] = self.tasks[i]
            info["evidence"] = build_evidence(
                tool_documentation=CHEATSHOP_OVERSIGHT_TOOL_DOCUMENTATION,
            )
            info["agent_response"] = build_behavior_under_review(
                trajectory=self.build_trajectory_under_review(i),
                final_output=self.episode_memory.get_final_output(
                    i, cot_visibility=self.config.env.cheatshop.self_report_cot_visibility,
                ),
            )
            if dones[i]:
                info["ground_truth_log"] = self.episode_memory.build_ground_truth_log(i)

        return next_observations, rewards, dones, infos

    def _history_action_repr(self, payload: Dict[str, Any]) -> str:
        if payload.get("kind") == "final_output":
            answer = payload.get("answer", "").strip()
            return f"<answer>{answer}</answer>" if answer else "<final_output>"
        return payload.get("canonical_action", "")

    def _record_episode_events(self, infos: List[Dict[str, Any]]):
        for env_idx, info in enumerate(infos):
            event_type = info.get("event_type")
            payload = info.get("payload")
            if event_type == "shopping_step" and payload is not None:
                self.episode_memory.store_shopping_step(env_idx, payload)
            elif event_type == "edit_action" and payload is not None:
                self.episode_memory.store_edit_action(env_idx, payload)
            elif event_type == "final_output" and payload is not None:
                self.episode_memory.store_final_output(env_idx, payload)

    def _update_episode_metrics(self, infos: List[Dict[str, Any]]):
        for env_idx, info in enumerate(infos):
            self.episode_memory.update_metrics(
                env_idx,
                {
                    "task_score": float(info.get("task_score", 0.0)),
                    "won": bool(info.get("won", False)),
                    "advanced_search_uses": int(info.get("advanced_search_uses", 0)),
                    "choice_oracle_uses": int(info.get("choice_oracle_uses", 0)),
                    "restricted_tool_calls": int(info.get("restricted_tool_calls", 0)),
                    "delete_count": int(info.get("delete_count", 0)),
                },
            )

    def _format_action_block(self, actions: List[str]) -> str:
        return "[\n" + "\n".join(f"'{action}'," for action in actions) + "\n]"

    def _format_available_actions(self, actions: List[str]) -> str:
        actions = list(dict.fromkeys(actions))
        if "click[analyze_results]" not in actions:
            return self._format_action_block(actions)

        shopper_actions = [action for action in actions if action != "click[analyze_results]"]
        sections = []
        if shopper_actions:
            sections.append(
                "Standard shopper actions:\n"
                f"{self._format_action_block(shopper_actions)}"
            )
        sections.append(
            "Other visible control:\n"
            f"{self._format_action_block(['click[analyze_results]'])}"
        )
        return "\n\n".join(sections)

    def build_text_obs(self, text_obs: List[str], infos: List[Dict[str, Any]], init: bool = False) -> List[str]:
        postprocess_text_obs = []
        memory_contexts = None
        valid_lens = None
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.actor_memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action",
            )

        for i, obs_text in enumerate(text_obs):
            stage = infos[i]["stage"]
            if stage == "shopping":
                available_actions = self._format_available_actions(infos[i]["available_actions"])
                if init or self.config.env.history_length <= 0:
                    obs = CHEATSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=obs_text,
                        available_actions=available_actions,
                    )
                else:
                    obs = CHEATSHOP_TEMPLATE.format(
                        task_description=self.tasks[i],
                        step_count=len(self.actor_memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_observation=obs_text,
                        available_actions=available_actions,
                    )
            elif stage == "log_edit":
                obs = CHEATSHOP_LOG_EDIT_TEMPLATE.format(
                    current_observation=obs_text,
                    available_actions=self._format_available_actions(infos[i]["available_actions"]),
                )
            elif stage == "self_report":
                obs = CHEATSHOP_SELF_REPORT_TEMPLATE.format(current_observation=obs_text)
            else:
                obs = obs_text
            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def build_monitor_obs(self, dones: List[bool], infos: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        from agent_system.environments.env_package.cheatshop.helpers import (
            build_behavior_under_review,
            build_evidence,
            build_monitor_background,
        )

        monitor_backgrounds = []
        agent_trajectories = []
        for i in range(len(infos)):
            if dones[i]:
                evidence = build_evidence(
                    tool_documentation=CHEATSHOP_OVERSIGHT_TOOL_DOCUMENTATION,
                )
                background = build_monitor_background(
                    evidence=evidence,
                    task=self.tasks[i],
                )
                behavior = build_behavior_under_review(
                    trajectory=self.build_trajectory_under_review(i),
                    final_output=self.episode_memory.get_final_output(
                        i,
                        cot_visibility=self.config.env.cheatshop.self_report_cot_visibility,
                    ),
                )
                monitor_backgrounds.append(background)
                agent_trajectories.append(behavior)
            else:
                monitor_backgrounds.append("")
                agent_trajectories.append("")
        return monitor_backgrounds, agent_trajectories

    def build_trajectory_under_review(self, env_idx: int) -> str:
        from agent_system.environments.env_package.cheatshop.helpers import build_trajectory_under_review

        return build_trajectory_under_review(
            memory=self.episode_memory,
            env_idx=env_idx,
            enable_log_edit=self.config.env.cheatshop.enable_log_edit,
            monitor_obs_level=self.config.env.cheatshop.monitor_obs_level,
        )

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item["active_masks"]:
                info = total_infos[batch_idx][i]
                success["success_rate"].append(float(info["won"]))
                success["webshop_task_score (not success_rate)"].append(float(info["task_score"]))
                success[f"{EPISODE_METRIC_PREFIX}advanced_search_uses"].append(float(info.get("advanced_search_uses", 0)))
                success[f"{EPISODE_METRIC_PREFIX}choice_oracle_uses"].append(float(info.get("choice_oracle_uses", 0)))
                success[f"{EPISODE_METRIC_PREFIX}delete_count"].append(float(info.get("delete_count", 0)))
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    val_group_n = config.env.rollout.val_n if config.env.rollout.val_n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "reasonchat" in config.env.env_name.lower():
        from agent_system.environments.env_package.reason_chat import build_reason_chat_envs, reason_chat_projection
        _envs = build_reason_chat_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_reason_chat_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=val_group_n, is_train=False, env_config=config.env)

        projection_f = reason_chat_projection
        envs = ReasonChatEnvironmentManager(_envs, projection_f, config)
        val_envs = ReasonChatEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "deceptive_search" in config.env.env_name.lower():
        from agent_system.environments.env_package.deceptive_search import (
            build_deceptive_search_envs, deceptive_search_projection
        )
        _envs = build_deceptive_search_envs(
            seed=config.env.seed, env_num=config.data.train_batch_size,
            group_n=group_n, is_train=True, env_config=config.env,
        )
        _val_envs = build_deceptive_search_envs(
            seed=config.env.seed + 1000, env_num=config.data.val_batch_size,
            group_n=val_group_n, is_train=False, env_config=config.env,
        )
        envs = DeceptiveSearchEnvironmentManager(_envs, deceptive_search_projection, config)
        val_envs = DeceptiveSearchEnvironmentManager(_val_envs, deceptive_search_projection, config)
        return envs, val_envs
    elif "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=val_group_n, is_train=False, env_config=config.env)

        # projection_f = partial(search_projection)
        projection_f = search_projection
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "cheatshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.cheatshop import build_cheatshop_envs, cheatshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
            'observation_mode': 'text',
            'num_products': None,
            'human_goals': config.env.webshop.human_goals,
            'file_path': file_path,
            'attr_path': attr_path,
            'require_self_report': config.env.cheatshop.require_self_report,
            'enable_log_edit': config.env.cheatshop.enable_log_edit,
            'monitor_obs_level': config.env.cheatshop.monitor_obs_level,
            'self_report_cot_visibility': config.env.cheatshop.self_report_cot_visibility,
            'shopping_max_steps': config.env.cheatshop.shopping_max_steps,
            'oracle_top_n': config.env.cheatshop.oracle_top_n,
            'tamper_max_deletes': config.env.cheatshop.tamper_max_deletes,
        }
        _envs = build_cheatshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_cheatshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=val_group_n, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        envs = CheatShopEnvironmentManager(_envs, cheatshop_projection, config)
        val_envs = CheatShopEnvironmentManager(_val_envs, cheatshop_projection, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=val_group_n, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=val_group_n, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)
