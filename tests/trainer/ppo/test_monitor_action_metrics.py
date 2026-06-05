import importlib.util
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

_REQUIRED_DEPS = ("torch",)
_MISSING_DEPS = [name for name in _REQUIRED_DEPS if importlib.util.find_spec(name) is None]
if _MISSING_DEPS:
    pytest.skip(
        "monitor metric utility test deps are not installed locally: " + ", ".join(_MISSING_DEPS),
        allow_module_level=True,
    )

import torch

from agent_system.utils.metric_contract import EPISODE_METRIC_PREFIX
from verl.trainer.ppo.metric_utils import (
    compute_monitor_action_metrics,
    compute_restricted_tool_confusion_metrics,
    dump_generations,
    dump_monitor_false_positives,
)


def _batch(non_tensor_batch, tensor_batch=None):
    return SimpleNamespace(
        non_tensor_batch=non_tensor_batch,
        batch=tensor_batch or {},
    )


def test_monitor_action_metrics():
    monitor_batch = _batch(
        {
            "monitor_action_type": np.array(["no_issue", "issue", "invalid", "no_issue"], dtype=object),
            "correct_no_issue": np.array([1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            "monitor_anchor_valid": np.array([-1.0, 1.0, -1.0, -1.0], dtype=np.float32),
            "monitor_link_valid": np.array([-1.0, 0.0, -1.0, -1.0], dtype=np.float32),
        }
    )

    metrics = compute_monitor_action_metrics(monitor_batch)

    assert metrics["monitor/action/no_issue_rate"] == pytest.approx(0.5)
    assert metrics["monitor/action/issue_rate"] == pytest.approx(0.25)
    assert metrics["monitor/action/invalid_rate"] == pytest.approx(0.25)
    assert metrics["monitor/action/correct_no_issue_rate"] == pytest.approx(0.5)
    assert metrics["monitor/action/anchor_pass_rate"] == pytest.approx(1.0)
    assert metrics["monitor/action/link_pass_rate"] == pytest.approx(0.0)


def test_restricted_tool_confusion_metrics_are_task_agnostic():
    actor_batch = _batch(
        {
            "traj_uid": np.array(["a", "b", "c", "d"], dtype=object),
            "trust_penalties": np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            f"{EPISODE_METRIC_PREFIX}used_insider": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            f"{EPISODE_METRIC_PREFIX}restricted_tool_calls": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        }
    )

    metrics = compute_restricted_tool_confusion_metrics(actor_batch, actor_cost_threshold=0.5)

    assert metrics["episode/monitor_confusion/tp_count"] == 1.0
    assert metrics["episode/monitor_confusion/fp_count"] == 1.0
    assert metrics["episode/monitor_confusion/fn_count"] == 1.0
    assert metrics["episode/monitor_confusion/tn_count"] == 1.0
    assert metrics["episode/restricted_tool_caught_rate"] == pytest.approx(0.5)
    assert metrics["episode/clean_false_positive_rate"] == pytest.approx(0.5)


def test_restricted_tool_confusion_metrics_skip_when_indicator_absent():
    actor_batch = _batch(
        {
            "traj_uid": np.array(["a"], dtype=object),
            "trust_penalties": np.array([1.0], dtype=np.float32),
        }
    )

    assert compute_restricted_tool_confusion_metrics(actor_batch, actor_cost_threshold=0.5) == {}


def test_dump_monitor_false_positive_samples(tmp_path):
    monitor_tokenizer = MagicMock()
    monitor_tokenizer.batch_decode.return_value = [
        "<issue><behavior_anchor>b</behavior_anchor><evidence_anchor>e</evidence_anchor><issue_relation>r</issue_relation></issue>",
        "<no_issue/>",
    ]
    actor_batch = _batch(
        {
            "traj_uid": np.array(["clean-fp", "restricted-tp"], dtype=object),
            "data_source": np.array(["search", "shop"], dtype=object),
            "episode_rewards": np.array([1.0, 0.5], dtype=np.float32),
            "trust_penalties": np.array([1.0, 1.0], dtype=np.float32),
            f"{EPISODE_METRIC_PREFIX}used_insider": np.array([0.0, 1.0], dtype=np.float32),
        }
    )
    monitor_batch = _batch(
        {
            "agent_trajectory": np.array(["behavior text", "restricted behavior"], dtype=object),
            "monitor_action_type": np.array(["issue", "issue"], dtype=object),
            "judge_score_token": np.array(["4", "4"], dtype=object),
        },
        tensor_batch={"responses": torch.zeros(2, 4, dtype=torch.long)},
    )

    dump_monitor_false_positives(
        actor_batch=actor_batch,
        monitor_batch=monitor_batch,
        dump_path=str(tmp_path),
        step=123,
        monitor_tokenizer=monitor_tokenizer,
        actor_cost_threshold=0.5,
    )

    output_path = tmp_path / "123.jsonl"
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0] == {
        "step": 123,
        "data_source": "search",
        "traj_uid": "clean-fp",
        "restricted_tool_used": False,
        "actor_trust_penalty": 1.0,
        "actor_cost_threshold": 0.5,
        "episode_reward": 1.0,
        "agent_trajectory": "behavior text",
        "monitor_output": "<issue><behavior_anchor>b</behavior_anchor><evidence_anchor>e</evidence_anchor><issue_relation>r</issue_relation></issue>",
        "monitor_action_type": "issue",
        "judge_score_token": "4",
    }


def test_dump_generations_samples(tmp_path):
    dump_generations(
        system_infos=["system"],
        inputs=["input"],
        outputs=["output"],
        scores=[0.75],
        reward_extra_infos_dict={"extra": ["value"], "wrong_len": []},
        dump_path=str(tmp_path),
        step=123,
    )

    rows = [json.loads(line) for line in (tmp_path / "123.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "system_info": "system",
            "input": "input",
            "output": "output",
            "score": 0.75,
            "step": 123,
            "extra": "value",
        }
    ]
