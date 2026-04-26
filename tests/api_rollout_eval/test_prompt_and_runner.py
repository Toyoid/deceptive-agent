import numpy as np
from omegaconf import OmegaConf

from agent_system.api_rollout_eval.clients import ChatResponse
from agent_system.api_rollout_eval.data import EvalRow
from agent_system.api_rollout_eval.prompt_builder import build_actor_messages
from agent_system.api_rollout_eval.runner import ApiRolloutRunner


class _PromptEnv:
    def actor_prompt_source(self, step):
        return "dataset" if step == 0 else "env_obs"


def test_prompt_builder_uses_dataset_prompt_on_bootstrap():
    row = EvalRow(
        data_source="reason_chat/test",
        prompt=[{"role": "system", "content": "sys"}, {"role": "user", "content": "question"}],
        env_kwargs={"question": "question"},
        index=0,
    )

    messages = build_actor_messages(
        row=row,
        envs=_PromptEnv(),
        obs={"text": ["env obs"]},
        infos=[{"system_prompt": "env sys"}],
        item=0,
        step=0,
    )

    assert messages == row.prompt


def test_prompt_builder_uses_env_observation_after_bootstrap():
    row = EvalRow(data_source="deceptive_search/test", prompt=[], env_kwargs={}, index=0)

    messages = build_actor_messages(
        row=row,
        envs=_PromptEnv(),
        obs={"text": ["search observation"]},
        infos=[{"system_prompt": "search sys", "format_prompt": "format"}],
        item=0,
        step=1,
    )

    assert messages == [
        {"role": "system", "content": "search sys\nformat"},
        {"role": "user", "content": "search observation"},
    ]


class _FakeClient:
    def __init__(self):
        self.calls = 0

    async def generate_batch(self, batch_messages):
        self.calls += 1
        return [
            ChatResponse(
                text=f"action-{self.calls}-{i}",
                finish_reason="stop",
                latency=0.1,
                total_tokens=10,
            )
            for i, _ in enumerate(batch_messages)
        ]


class _FakeEnv:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.step_count = 0

    def reset(self, kwargs):
        infos = [{"system_prompt": "sys", "won": False, "task_score": 0.0} for _ in range(self.batch_size)]
        return {"text": [f"obs-{i}" for i in range(self.batch_size)], "image": None, "anchor": None}, infos

    def actor_prompt_source(self, step):
        return "dataset" if step == 0 else "env_obs"

    def get_rollout_max_steps(self):
        return 2

    def step(self, text_actions):
        self.step_count += 1
        done = self.step_count >= 2
        infos = []
        for i, action in enumerate(text_actions):
            infos.append(
                {
                    "won": done,
                    "task_score": float(done),
                    "is_action_valid": not action.endswith("-1"),
                    "tool_calling": 1.0,
                    "agent_response": action,
                }
            )
        return (
            {"text": [f"next-{self.step_count}-{i}" for i in range(self.batch_size)], "image": None, "anchor": None},
            np.ones(self.batch_size, dtype=np.float32),
            np.asarray([done] * self.batch_size),
            infos,
        )

    def success_evaluator(self, **kwargs):
        total_batch_list = kwargs["total_batch_list"]
        return {
            "success_rate": np.ones(len(total_batch_list), dtype=np.float32),
            "episode_metric/delete_count": np.arange(len(total_batch_list), dtype=np.float32),
        }

    def close(self):
        pass


def test_runner_collects_actor_only_episodes_and_metrics():
    rows = [
        EvalRow(
            data_source="fake",
            prompt=[{"role": "user", "content": f"question-{i}"}],
            env_kwargs={"question": f"question-{i}"},
            index=i,
            ability="search",
            reward_model={"ground_truth": f"answer-{i}"},
            extra_info={"index": i},
            metadata={"split": "test"},
            source_file="fake.parquet",
        )
        for i in range(2)
    ]
    cfg = OmegaConf.create({"data": {"batch_size": 2}, "env": {"max_steps": 2}})
    runner = ApiRolloutRunner(
        config=cfg,
        client=_FakeClient(),
        rows=rows,
        env_factory=lambda config, batch_size: _FakeEnv(batch_size),
    )

    result = runner.run()

    assert len(result.episodes) == 2
    assert result.episodes[0]["length"] == 2.0
    assert result.episodes[0]["final_output"] == "action-2-0"
    assert result.episodes[0]["ability"] == "search"
    assert result.episodes[0]["reward_model"] == {"ground_truth": "answer-0"}
    assert result.episodes[0]["extra_info"] == {"index": 0}
    assert result.episodes[0]["metadata"] == {"split": "test"}
    assert result.episodes[0]["source_file"] == "fake.parquet"
    assert result.episodes[1]["action_valid_sequence"] == [False, False]
    assert result.metrics["eval/success_rate"] == 1.0
    assert result.metrics["eval/episode/delete_count/max"] == 1.0
