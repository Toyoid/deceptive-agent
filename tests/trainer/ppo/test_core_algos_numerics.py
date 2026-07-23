# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Regression tests for finite PPO gradients at extreme log-ratios."""

import pytest


torch = pytest.importorskip("torch")
core_algos = pytest.importorskip("verl.trainer.ppo.core_algos")


def test_clipped_policy_loss_has_finite_gradient_for_extreme_ratio():
    old_log_prob = torch.zeros(1, 1)
    log_prob = torch.tensor([[1000.0]], requires_grad=True)
    advantages = torch.ones(1, 1)
    response_mask = torch.ones(1, 1)

    pg_loss, _, _, _ = core_algos.compute_policy_loss(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        cliprange=0.2,
    )
    pg_loss.backward()

    assert torch.isfinite(pg_loss)
    assert torch.isfinite(log_prob.grad).all()


def test_low_variance_kl_has_finite_gradient_for_extreme_ratio():
    log_prob = torch.tensor([[-1000.0]], requires_grad=True)
    ref_log_prob = torch.zeros(1, 1)

    loss = core_algos.kl_penalty(log_prob, ref_log_prob, "low_var_kl").sum()
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(log_prob.grad).all()
