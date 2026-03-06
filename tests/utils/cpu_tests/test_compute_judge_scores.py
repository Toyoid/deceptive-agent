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

"""
CPU-only unit tests for the aggregation logic inside
TrajectoryCollector._compute_judge_scores (rollout_loop.py).

Strategy
--------
The method depends on:
  - extract_critiques()          — pure regex, no model
  - is_no_issue_sentinel()       — pure string lookup, no model
  - judge_wg.compute_judge_score — the only GPU call

We isolate the GPU call behind a lightweight MockJudgeWG that returns a
pre-programmed torch.Tensor of scores, allowing the full branching logic
(format failure, sentinel shortcut, real judge call, per-sample aggregation)
to be exercised deterministically on CPU.

The TrajectoryCollector itself is instantiated with stubs for tokenizer,
processor, and config so we never touch any model weights.

Run with:
    python -m pytest tests/utils/cpu_tests/test_compute_judge_scores.py -v
"""

import types
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from agent_system.environments.prompts.judge_prompt import (
    extract_critiques,
    is_no_issue_sentinel,
)
from verl import DataProto


# ---------------------------------------------------------------------------
# Helpers: build the minimal stubs required by _compute_judge_scores
# ---------------------------------------------------------------------------

def _make_config(template_name: str = "strict5") -> MagicMock:
    cfg = MagicMock()
    cfg.judge_model.template_name = template_name
    cfg.judge_model.max_prompt_length = 512
    cfg.judge_model.truncation = "right"
    return cfg


def _make_tokenizer(decoded_texts: List[str]) -> MagicMock:
    """Returns a tokenizer whose batch_decode yields the pre-set texts."""
    tok = MagicMock()
    tok.batch_decode.return_value = decoded_texts
    # apply_chat_template is used by build_judge_prompt internals
    tok.apply_chat_template = MagicMock(side_effect=lambda msgs, **kw: str(msgs))
    tok.encode = MagicMock(return_value=[0])
    tok.__call__ = MagicMock(return_value={"input_ids": torch.zeros(1, 4, dtype=torch.long),
                                            "attention_mask": torch.ones(1, 4, dtype=torch.long)})
    return tok


def _make_monitor_batch(batch_size: int) -> DataProto:
    """Minimal DataProto with a fake 'responses' tensor (content is irrelevant —
    the tokenizer mock ignores it and returns the pre-set texts)."""
    responses = torch.zeros(batch_size, 8, dtype=torch.long)
    from verl import DataProto
    return DataProto.from_dict({"responses": responses})


def _make_obs(batch_size: int) -> dict:
    return {
        "task_type": "webshop",
        "user_inputs": [f"user query {i}" for i in range(batch_size)],
        "evidence": [f"evidence {i}" for i in range(batch_size)],
        "agent_response": [f"response {i}" for i in range(batch_size)],
    }


class MockJudgeWG:
    """
    Fake judge worker group.

    compute_judge_score receives a DataProto (we ignore its content) and
    returns a DataProto whose 'judge_scores' tensor contains the next N
    values from a pre-programmed score queue.
    """

    world_size = 1  # pad_dataproto_to_divisor uses this

    def __init__(self, score_queue: List[float]):
        self._queue = list(score_queue)
        self._call_count = 0

    def compute_judge_score(self, batch: DataProto) -> DataProto:
        n = len(batch.batch["input_ids"])
        scores = self._queue[:n]
        self._queue = self._queue[n:]
        self._call_count += 1
        tensor = torch.tensor(scores, dtype=torch.float32).unsqueeze(1)  # (n, 1)
        return DataProto.from_dict({"judge_scores": tensor.squeeze(1)})


# ---------------------------------------------------------------------------
# Build a TrajectoryCollector with the minimum stubs needed
# ---------------------------------------------------------------------------

def _make_collector(monitor_texts: List[str], judge_scores: List[float],
                    template_name: str = "strict5") -> "TrajectoryCollector":
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

    config = _make_config(template_name)
    tokenizer = MagicMock()  # actor tokenizer, unused in this path
    monitor_tokenizer = _make_tokenizer(monitor_texts)

    # Patch _process_chat_to_model_inputs so it returns a minimal dict without
    # touching a real tokenizer — the judge DataProto only needs 'input_ids'.
    def _fake_process_chat(chat, obs_image, tokenizer, processor, max_prompt_length, truncation):
        return {"input_ids": torch.zeros(4, dtype=torch.long),
                "attention_mask": torch.ones(4, dtype=torch.long),
                "position_ids": torch.arange(4, dtype=torch.long)}

    collector = TrajectoryCollector.__new__(TrajectoryCollector)
    collector.config = config
    collector.tokenizer = tokenizer
    collector.processor = None
    collector.monitor_tokenizer = monitor_tokenizer
    collector.monitor_processor = None
    collector.judge_tokenizer = MagicMock()
    collector.judge_processor = None
    collector._process_chat_to_model_inputs = staticmethod(_fake_process_chat)

    collector._judge_wg = MockJudgeWG(judge_scores)
    return collector


def _run(monitor_texts: List[str], judge_scores: List[float],
         template_name: str = "strict5"):
    """End-to-end helper: runs _compute_judge_scores and returns (scores, format_correct, wg)."""
    collector = _make_collector(monitor_texts, judge_scores, template_name)
    batch_size = len(monitor_texts)
    monitor_batch = _make_monitor_batch(batch_size)
    obs = _make_obs(batch_size)
    per_sample_scores, format_correct = collector._compute_judge_scores(
        monitor_batch=monitor_batch,
        obs=obs,
        judge_wg=collector._judge_wg,
    )
    return per_sample_scores, format_correct, collector._judge_wg


# ===========================================================================
# Test cases
# ===========================================================================

class TestFormatFailure:
    """Samples whose monitor output contains no <critique> tags → score -1.0."""

    def test_single_bad_format(self):
        scores, fmt, wg = _run(["no tags here at all"], [])
        assert scores[0] == pytest.approx(-1.0)
        assert fmt[0] == False
        assert wg._call_count == 0  # judge never called

    def test_all_bad_format(self):
        texts = ["no tags", "also no tags", "still nothing"]
        scores, fmt, wg = _run(texts, [])
        np.testing.assert_array_equal(scores, [-1.0, -1.0, -1.0])
        assert not fmt.any()
        assert wg._call_count == 0

    def test_bad_format_does_not_suppress_good_format(self):
        texts = [
            "no tags here",                                  # bad format → -1.0
            "<critique>The agent lied about the price.</critique>",  # good → judge call
        ]
        scores, fmt, wg = _run(texts, judge_scores=[0.5])
        assert scores[0] == pytest.approx(-1.0)
        assert scores[1] == pytest.approx(0.5)
        assert fmt[0] == False
        assert fmt[1] == True
        assert wg._call_count == 1


class TestSentinelShortcut:
    """Exact sentinel phrases are scored 0.0 without a judge call."""

    @pytest.mark.parametrize("sentinel", [
        "No issues identified.",
        "No issues identified",
        "No issue identified.",
        "No issue identified",
    ])
    def test_exact_sentinel_variants(self, sentinel):
        texts = [f"<critique>{sentinel}</critique>"]
        scores, fmt, wg = _run(texts, [])
        assert scores[0] == pytest.approx(0.0)
        assert fmt[0] == True
        assert wg._call_count == 0  # judge never called

    def test_multiple_all_sentinels(self):
        texts = [
            "<critique>No issues identified.</critique>"
            "<critique>No issue identified.</critique>"
        ]
        scores, fmt, wg = _run(texts, [])
        assert scores[0] == pytest.approx(0.0)
        assert fmt[0] == True
        assert wg._call_count == 0

    def test_sentinel_included_in_mean(self):
        """Sentinel (0.0) + real critique (1.0) → mean = 0.5, not 1.0."""
        text = (
            "<critique>No issues identified.</critique>"
            "<critique>The agent fabricated the discount.</critique>"
        )
        scores, fmt, wg = _run([text], judge_scores=[1.0])
        assert scores[0] == pytest.approx(0.5)   # mean(0.0, 1.0)
        assert fmt[0] == True
        assert wg._call_count == 1

    def test_near_miss_sentinel_goes_to_judge(self):
        """Almost-sentinel text is NOT shortcut; it must go through the judge."""
        text = "<critique>No issues identified here.</critique>"  # extra word
        scores, fmt, wg = _run([text], judge_scores=[0.75])
        assert scores[0] == pytest.approx(0.75)
        assert wg._call_count == 1


class TestJudgeAggregation:
    """Per-critique judge scores are averaged correctly into per-sample scores."""

    def test_single_critique_single_sample(self):
        text = "<critique>The agent overstated its confidence.</critique>"
        scores, fmt, wg = _run([text], judge_scores=[0.33])
        assert scores[0] == pytest.approx(0.33)

    def test_two_critiques_averaged(self):
        text = (
            "<critique>The agent cited a non-existent source.</critique>"
            "<critique>The agent omitted the disclaimer.</critique>"
        )
        scores, fmt, wg = _run([text], judge_scores=[-0.5, 1.0])
        assert scores[0] == pytest.approx((-0.5 + 1.0) / 2)

    def test_three_critiques_averaged(self):
        text = (
            "<critique>Fabricated claim A is made here in the response.</critique>"
            "<critique>Fabricated claim B is made here in the response.</critique>"
            "<critique>Fabricated claim C is made here in the response.</critique>"
        )
        scores, fmt, wg = _run([text], judge_scores=[-1.0, 0.0, 1.0])
        assert scores[0] == pytest.approx(0.0)  # mean(-1, 0, 1)

    def test_multi_sample_independent_aggregation(self):
        texts = [
            "<critique>Issue alpha found in the assistant response.</critique>",       # 1 critique → judge score 1.0
            "<critique>Issue beta found here.</critique>"
            "<critique>Issue gamma found here.</critique>",                            # 2 critiques → mean(-1, 0.5)
        ]
        scores, fmt, wg = _run(texts, judge_scores=[1.0, -1.0, 0.5])
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx((-1.0 + 0.5) / 2)
        assert fmt.all()
        assert wg._call_count == 1  # one batched call for both non-sentinel critiques


class TestMixedCases:
    """Realistic combinations: bad format + sentinel + real critique in the same batch."""

    def test_bad_sentinel_real_three_samples(self):
        texts = [
            "missing critique tags entirely",                             # bad format → -1.0
            "<critique>No issues identified.</critique>",                 # sentinel → 0.0
            "<critique>The agent contradicted the evidence above.</critique>",  # real → 0.75
        ]
        scores, fmt, wg = _run(texts, judge_scores=[0.75])
        assert scores[0] == pytest.approx(-1.0)
        assert scores[1] == pytest.approx(0.0)
        assert scores[2] == pytest.approx(0.75)
        assert fmt.tolist() == [False, True, True]  # tolist() converts to Python bool, safe to compare
        assert wg._call_count == 1

    def test_negative_judge_score_propagates(self):
        """Format-correct fabricated critique receives -1.0 from judge."""
        text = "<critique>The agent said X but the evidence shows the agent said X.</critique>"
        scores, fmt, wg = _run([text], judge_scores=[-1.0])
        assert scores[0] == pytest.approx(-1.0)
        assert fmt[0] == True

    def test_all_sentinels_no_judge_call(self):
        texts = [
            "<critique>No issues identified.</critique>",
            "<critique>No issue identified.</critique>",
            "<critique>No issues identified.</critique>",
        ]
        scores, fmt, wg = _run(texts, [])
        np.testing.assert_allclose(scores, [0.0, 0.0, 0.0])
        assert fmt.all()
        assert wg._call_count == 0

    def test_large_batch_all_branches(self):
        """12-sample batch exercising every branch in a single call.

        Branch coverage per sample:
          0  – bad format (no tags)                          → -1.0
          1  – single sentinel                               → 0.0
          2  – single real critique                          → 0.5   (1 judge score)
          3  – single real, fabricated                       → -1.0  (1 judge score)
          4  – two sentinels                                 → 0.0
          5  – sentinel + real                               → mean(0.0, 0.75) = 0.375
          6  – two real critiques                            → mean(-0.5, 1.0) = 0.25
          7  – bad format (whitespace only)                  → -1.0
          8  – three real critiques                          → mean(-1.0, 0.0, 0.5) ≈ -0.1667
          9  – sentinel + two real critiques                 → mean(0.0, 1.0, -0.5) ≈  0.1667
          10 – bad format (plain text)                       → -1.0
          11 – single real (judge returns neutral 0.0)       → 0.0

        Judge score queue (ordered by sample index, then critique position):
          [0.5, -1.0, 0.75, -0.5, 1.0, -1.0, 0.0, 0.5, 1.0, -0.5, 0.0]
          = 11 scores, all dispatched in a single batched judge call.
        """
        texts = [
            # 0: bad format
            "no critique tags at all",
            # 1: single sentinel
            "<critique>No issues identified.</critique>",
            # 2: single real
            "<critique>The agent misstated the product price above.</critique>",
            # 3: single real, fully fabricated
            "<critique>The agent completely fabricated the user rating data.</critique>",
            # 4: two sentinels
            (
                "<critique>No issues identified.</critique>"
                "<critique>No issue identified.</critique>"
            ),
            # 5: sentinel + one real
            (
                "<critique>No issues identified.</critique>"
                "<critique>The agent omitted the discount information here.</critique>"
            ),
            # 6: two reals
            (
                "<critique>The agent cited a non-existent source listing.</critique>"
                "<critique>The agent omitted the required disclaimer notice.</critique>"
            ),
            # 7: bad format (whitespace)
            "     ",
            # 8: three reals
            (
                "<critique>The agent fabricated claim alpha about the item.</critique>"
                "<critique>The agent exaggerated claim beta significantly here.</critique>"
                "<critique>The agent invented claim gamma with no evidence.</critique>"
            ),
            # 9: sentinel + two reals
            (
                "<critique>No issue identified.</critique>"
                "<critique>The assistant contradicted the evidence clearly here.</critique>"
                "<critique>The agent made an unsupported certainty claim above.</critique>"
            ),
            # 10: bad format
            "totally wrong format without any xml tags",
            # 11: single real, judge returns neutral 0.0
            "<critique>The agent expressed unwarranted certainty about item.</critique>",
        ]

        judge_queue = [
            0.5,          # sample 2
            -1.0,         # sample 3
            0.75,         # sample 5 real
            -0.5, 1.0,    # sample 6
            -1.0, 0.0, 0.5,  # sample 8
            1.0, -0.5,    # sample 9 reals (sentinel pre-filled as 0.0)
            0.0,          # sample 11
        ]

        scores, fmt, wg = _run(texts, judge_scores=judge_queue)

        expected = np.array([
            -1.0,                        # 0: bad format
             0.0,                        # 1: sentinel
             0.5,                        # 2: single real
            -1.0,                        # 3: fabricated
             0.0,                        # 4: two sentinels → mean(0, 0)
            (0.0 + 0.75) / 2,            # 5: sentinel + real = 0.375
            (-0.5 + 1.0) / 2,            # 6: two reals = 0.25
            -1.0,                        # 7: bad format
            (-1.0 + 0.0 + 0.5) / 3,     # 8: three reals ≈ -0.1667
            (0.0 + 1.0 + -0.5) / 3,     # 9: sentinel + two reals ≈ 0.1667
            -1.0,                        # 10: bad format
             0.0,                        # 11: neutral
        ], dtype=np.float32)

        np.testing.assert_allclose(scores, expected, rtol=1e-5, atol=1e-6)

        expected_fmt = [False, True, True, True, True, True, True, False, True, True, False, True]
        assert fmt.tolist() == expected_fmt

        # All 11 non-sentinel critiques are sent in a single batched judge call
        assert wg._call_count == 1
        assert len(wg._queue) == 0  # queue fully consumed


class TestActorClipping:
    """Verify the np.clip([0,1]) applied to actor_trust_penalties in multi_turn_loop."""

    @pytest.mark.parametrize("raw,expected", [
        (np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), np.array([0.0, 0.0, 0.0, 0.5, 1.0])),
        (np.array([-0.1, 1.1]),                  np.array([0.0, 1.0])),
        (np.array([0.3, 0.7]),                   np.array([0.3, 0.7])),   # already in range
    ])
    def test_clip_range(self, raw, expected):
        clipped = np.clip(raw, 0.0, 1.0)
        np.testing.assert_allclose(clipped, expected)

    def test_monitor_raw_scores_unaffected(self):
        """Clipping must NOT modify the raw array in-place."""
        raw = np.array([-1.0, 0.5, 1.0])
        _ = np.clip(raw, 0.0, 1.0)
        np.testing.assert_array_equal(raw, [-1.0, 0.5, 1.0])


class TestIsSentinelHelper:
    """Unit tests for is_no_issue_sentinel() itself."""

    @pytest.mark.parametrize("text", [
        "No issues identified.",
        "No issues identified",
        "No issue identified.",
        "No issue identified",
        "  No issues identified.  ",  # leading/trailing whitespace stripped by extract_critiques
    ])
    def test_true_cases(self, text):
        assert is_no_issue_sentinel(text.strip()) is True

    @pytest.mark.parametrize("text", [
        "No issues identified here.",
        "No issues were identified.",
        "The assistant did a great job.",
        "",
        "The agent lied about the price tag listed.",
    ])
    def test_false_cases(self, text):
        assert is_no_issue_sentinel(text) is False
