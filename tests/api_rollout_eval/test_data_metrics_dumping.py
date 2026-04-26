import json
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from agent_system.api_rollout_eval.data import load_eval_rows
from agent_system.api_rollout_eval.dumping import build_generation_samples, dump_trajectories, select_episodes
from agent_system.api_rollout_eval.metrics import compute_eval_metrics


def _config(**overrides):
    base = {
        "data": {
            "files": None,
            "num_episodes": None,
            "batch_size": 2,
            "max_samples": None,
            "shuffle": False,
            "seed": 123,
        },
        "env": {"env_name": "CheatShop", "rollout": {"n": 1}},
        "dump": {
            "output_dir": None,
            "n_trajectories": 2,
            "policy": "first",
            "include_steps": True,
            "log_generations": 2,
        },
    }
    for section, values in overrides.items():
        base[section].update(values)
    return OmegaConf.create(base)


def test_cheatshop_synthesizes_rows_and_ignores_files():
    cfg = _config(data={"files": ["ignored.parquet"], "num_episodes": 3})

    rows = load_eval_rows(cfg)

    assert len(rows) == 3
    assert all(row.data_source == "cheatshop" for row in rows)
    assert all(row.env_kwargs is None for row in rows)


def test_cheatshop_requires_num_episodes():
    with pytest.raises(ValueError, match="num_episodes"):
        load_eval_rows(_config())


def test_parquet_loader_uses_datasets_copy_to_local_and_preserves_metadata(monkeypatch):
    records = [
        {
            "data_source": "reason_chat/test",
            "prompt": [{"role": "user", "content": "hello"}],
            "env_kwargs": {"question": "hello"},
            "ability": "alignment",
            "reward_model": {"style": "rm"},
            "extra_info": {"index": 17, "split": "test"},
            "metadata": {"source": "unit"},
        }
    ]
    load_calls = []
    copy_calls = []

    class FakeDataset(list):
        @property
        def column_names(self):
            return list(records[0].keys())

    def fake_load_dataset(fmt, data_files):
        load_calls.append((fmt, data_files))
        return {"train": FakeDataset(records)}

    def fake_copy_to_local(src, cache_dir=None, use_shm=False):
        copy_calls.append((src, cache_dir, use_shm))
        return f"local::{src}"

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    monkeypatch.setitem(sys.modules, "verl.utils.fs", types.SimpleNamespace(copy_to_local=fake_copy_to_local))

    cfg = _config(
        data={"files": ["remote.parquet"], "cache_dir": "cache", "use_shm": True},
        env={"env_name": "ReasonChat"},
    )
    rows = load_eval_rows(cfg)

    assert copy_calls == [("remote.parquet", "cache", True)]
    assert load_calls == [("parquet", "local::remote.parquet")]
    assert len(rows) == 1
    assert rows[0].index == 17
    assert rows[0].ability == "alignment"
    assert rows[0].reward_model == {"style": "rm"}
    assert rows[0].extra_info == {"index": 17, "split": "test"}
    assert rows[0].metadata == {"source": "unit"}


def test_metrics_aggregate_episode_api_and_env_metrics():
    episodes = [
        {
            "reward": 1.0,
            "length": 2,
            "tool_calls": 1,
            "data_source": "source_a",
            "action_valid_sequence": [True, False],
            "steps": [{"api_latency": 0.1, "api_error": None, "total_tokens": 5}],
        },
        {
            "reward": 3.0,
            "length": 4,
            "tool_calls": 2,
            "data_source": "source_a",
            "action_valid_sequence": [True, True],
            "steps": [{"api_latency": 0.3, "api_error": "boom", "total_tokens": 7}],
        },
    ]
    success = {
        "success_rate": np.asarray([1.0, 0.0]),
        "answer_correct_rate": np.asarray([1.0, 1.0]),
        "episode_metric/delete_count": np.asarray([0.0, 2.0]),
    }

    metrics = compute_eval_metrics(episodes=episodes, success=success)

    assert metrics["eval/episode/reward/mean"] == 2.0
    assert metrics["eval/action_valid_rate"] == 0.75
    assert metrics["eval/api/error_rate"] == 0.5
    assert metrics["eval/success_rate"] == 0.5
    assert metrics["eval/answer_correct_rate"] == 1.0
    assert metrics["eval/episode/delete_count/max"] == 2.0


def test_dumping_caps_written_trajectories_without_changing_inputs():
    tmp_path = Path.cwd() / f"api_rollout_eval_dump_test_{os.getpid()}"
    tmp_path.mkdir(exist_ok=True)
    episodes = [
        {
            "trajectory_id": f"traj-{i}",
            "initial_input": f"in-{i}",
            "final_output": f"out-{i}",
            "reward": float(i),
            "steps": [{"step": 0, "value": i}],
        }
        for i in range(4)
    ]
    cfg = _config(dump={"output_dir": str(tmp_path), "n_trajectories": 2, "policy": "first"})

    try:
        written = dump_trajectories(episodes, cfg)

        assert set(written) == {"episodes", "steps"}
        assert len(episodes) == 4
        episode_lines = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
        step_lines = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
        assert [line["trajectory_id"] for line in episode_lines] == ["traj-0", "traj-1"]
        assert [line["trajectory_id"] for line in step_lines] == ["traj-0", "traj-1"]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_random_selection_is_deterministic():
    episodes = [{"trajectory_id": str(i)} for i in range(10)]

    first = select_episodes(episodes, n_trajectories=4, policy="random", seed=7)
    second = select_episodes(episodes, n_trajectories=4, policy="random", seed=7)

    assert [item["trajectory_id"] for item in first] == [item["trajectory_id"] for item in second]


def test_generation_samples_use_dump_policy():
    episodes = [
        {"initial_input": "b", "final_output": "B", "reward": 2.0},
        {"initial_input": "a", "final_output": "A", "reward": 1.0},
    ]
    cfg = _config(dump={"log_generations": 1, "policy": "first"})

    assert build_generation_samples(episodes, cfg) == [("b", "B", 2.0)]
