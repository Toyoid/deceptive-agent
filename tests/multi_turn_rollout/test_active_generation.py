import numpy as np
import pytest
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from verl import DataProto


class _FakeTokenizer:
    def batch_decode(self, responses, skip_special_tokens=True):
        del skip_special_tokens
        return [f"action-{int(row[0].item())}" for row in responses]


class _FakeActorRolloutWorker:
    world_size = 1

    def __init__(self):
        self.calls = 0
        self.batch_sizes = []

    def generate_sequences(self, batch):
        self.calls += 1
        batch_size = len(batch)
        self.batch_sizes.append(batch_size)
        responses = torch.arange(batch_size, dtype=torch.long).unsqueeze(1) + self.calls * 10
        prompts = batch.batch["input_ids"]
        seq = torch.cat([prompts, responses], dim=-1)
        response_mask = torch.ones_like(responses)
        return DataProto.from_dict(
            tensors={
                "prompts": prompts,
                "responses": responses,
                "input_ids": seq,
                "rollout_log_probs": torch.zeros_like(responses, dtype=torch.float32),
                "attention_mask": torch.cat([batch.batch["attention_mask"], response_mask], dim=-1),
                "position_ids": torch.cat([batch.batch["position_ids"], batch.batch["position_ids"][:, -1:] + 1], dim=-1),
            }
        )


class _StaggeredDoneTrainingEnv:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.step_count = 0
        self.step_actions = []

    def reset(self, kwargs):
        assert len(kwargs) == self.batch_size
        infos = [{"won": False, "user_input": f"user-{i}", "evidence": f"evidence-{i}"} for i in range(self.batch_size)]
        return {"text": None, "image": None, "anchor": None}, infos

    def actor_prompt_source(self, step):
        return "dataset"

    def get_rollout_max_steps(self):
        return 3

    def step(self, text_actions):
        self.step_count += 1
        self.step_actions.append(list(text_actions))
        dones = np.asarray([self.step_count >= end_step for end_step in (1, 2, 3)], dtype=bool)
        infos = []
        for i, action in enumerate(text_actions):
            infos.append(
                {
                    "won": bool(dones[i]),
                    "is_action_valid": True,
                    "tool_calling": 1.0,
                    "user_input": f"user-{i}",
                    "evidence": f"evidence-{i}",
                    "agent_response": action,
                }
            )
        return {"text": None, "image": None, "anchor": None}, np.ones(self.batch_size, dtype=np.float32), dones, infos

    def success_evaluator(self, **kwargs):
        total_batch_list = kwargs["total_batch_list"]
        return {"success_rate": np.asarray([len(items) > 0 for items in total_batch_list], dtype=np.float32)}


class _FlakyDoneTrainingEnv(_StaggeredDoneTrainingEnv):
    def get_rollout_max_steps(self):
        return 5

    def step(self, text_actions):
        self.step_count += 1
        self.step_actions.append(list(text_actions))
        dones = np.asarray([self.step_count == 1, self.step_count == 4], dtype=bool)
        infos = []
        for i, action in enumerate(text_actions):
            infos.append(
                {
                    "won": bool(dones[i]),
                    "is_action_valid": True,
                    "tool_calling": 1.0,
                    "user_input": f"user-{i}",
                    "evidence": f"evidence-{i}",
                    "agent_response": action,
                }
            )
        return {"text": None, "image": None, "anchor": None}, np.ones(self.batch_size, dtype=np.float32), dones, infos


def _make_gen_batch(batch_size):
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(batch_size * 2, dtype=torch.long).reshape(batch_size, 2),
            "attention_mask": torch.ones(batch_size, 2, dtype=torch.long),
            "position_ids": torch.arange(batch_size * 2, dtype=torch.long).reshape(batch_size, 2),
        },
        non_tensors={
            "raw_prompt_ids": np.asarray([[i, i + 1] for i in range(batch_size)], dtype=object),
            "data_source": np.asarray(["fake"] * batch_size, dtype=object),
            "env_kwargs": np.asarray([{"idx": i} for i in range(batch_size)], dtype=object),
        },
        meta_info={},
    )


def test_training_rollout_generates_active_only_but_steps_full_batch():
    batch_size = 3
    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    collector.config = OmegaConf.create(
        {
            "self_monitor": {"enable": False},
            "monitor_rollout_ref": {"enable": False},
            "verdict_monitor": {"enable": False},
            "judge_model": {"enable": False},
        }
    )
    collector.tokenizer = _FakeTokenizer()

    actor = _FakeActorRolloutWorker()
    env = _StaggeredDoneTrainingEnv(batch_size)

    actor_batch_dict, monitor_batch = collector.vanilla_multi_turn_loop(
        gen_batch=_make_gen_batch(batch_size),
        actor_rollout_wg=actor,
        monitor_wg=None,
        verdict_monitor_wg=None,
        judge_wg=None,
        envs=env,
        rollout_n=1,
        monitor_rollout_n=1,
    )

    assert monitor_batch is None
    assert actor.batch_sizes == [3, 2, 1]
    assert env.step_actions == [
        ["action-10", "action-11", "action-12"],
        ["action-10", "action-20", "action-21"],
        ["action-10", "action-20", "action-30"],
    ]
    assert actor_batch_dict["episode_lengths"].tolist() == [1.0, 2.0, 3.0]
    assert [len(items) for items in actor_batch_dict["total_batch_list"]] == [1, 2, 3]
    assert all(record["active_masks"] for items in actor_batch_dict["total_batch_list"] for record in items)


def test_training_rollout_done_slots_do_not_reactivate_if_env_done_flickers_false():
    batch_size = 2
    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    collector.config = OmegaConf.create(
        {
            "self_monitor": {"enable": False},
            "monitor_rollout_ref": {"enable": False},
            "verdict_monitor": {"enable": False},
            "judge_model": {"enable": False},
        }
    )
    collector.tokenizer = _FakeTokenizer()

    actor = _FakeActorRolloutWorker()
    env = _FlakyDoneTrainingEnv(batch_size)

    actor_batch_dict, monitor_batch = collector.vanilla_multi_turn_loop(
        gen_batch=_make_gen_batch(batch_size),
        actor_rollout_wg=actor,
        monitor_wg=None,
        verdict_monitor_wg=None,
        judge_wg=None,
        envs=env,
        rollout_n=1,
        monitor_rollout_n=1,
    )

    assert monitor_batch is None
    assert actor.batch_sizes == [2, 1, 1, 1]
    assert env.step_actions == [
        ["action-10", "action-11"],
        ["action-10", "action-20"],
        ["action-10", "action-30"],
        ["action-10", "action-40"],
    ]
    assert actor_batch_dict["episode_lengths"].tolist() == [1.0, 4.0]
    assert [len(items) for items in actor_batch_dict["total_batch_list"]] == [1, 4]
