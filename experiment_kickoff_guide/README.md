# Experiment Kickoff Guide

This directory is the developer-facing experiment manual for the deception and misalignment experiments in this repository. It supplements the root [README.md](../README.md), which introduces the repository, and [README_legacy_verl_agent.md](../README_legacy_verl_agent.md), which remains the authoritative reference for inherited Search and WebShop environment setup details.

Use these pages in order:

1. [quick_start.md](quick_start.md): concise launch steps for all supported environment and method combinations. This page assumes environments are already installed.
2. [roles.md](roles.md): ReasonChat/deceptive-roles environment introduction, config notes, installation assumptions, data preparation, launches, and API rollout evaluation.
3. [deceptive_search.md](deceptive_search.md): Deceptive Search environment introduction, Search-R1 setup, retriever requirements, data preparation, launches, and API rollout evaluation.
4. [cheatshop.md](cheatshop.md): CheatShop environment introduction, WebShop/CheatShop setup, data preparation, launches, and API rollout evaluation.
5. [methods.md](methods.md): RL, self-monitor, verdict-monitor, auxiliary safety RLHF, critique-monitor, maximin, trust-penalty application, and judge configuration.

## Experiment Scope

The guide covers three target environments and six method families.

| Environment | `env.env_name` | Main page | Notes |
| --- | --- | --- | --- |
| roles | `ReasonChat` | [roles.md](roles.md) | Single-turn deceptive role chat. Reward comes from a reward model, so Lagrangian trust-penalty application is usually preferred. |
| deceptive search | `deceptive_search` | [deceptive_search.md](deceptive_search.md) | Search-R1-style QA with a public search tool and hidden insider lookup. Requires the retriever service. Mitigation runs start from a prior RL checkpoint. |
| cheatshop | `CheatShop` | [cheatshop.md](cheatshop.md) | WebShop-derived shopping task with restricted/internal tools and monitor-visible traces. Runs in `deceptive-agent-webshop`. Mitigation runs start from a prior RL checkpoint. |

| Method | Status in this guide | Main configs |
| --- | --- | --- |
| RL | Baseline task RL and checkpoint producer for deceptive search and cheatshop mitigation. | `algorithm.adv_estimator`, `actor_rollout_ref.*`, `env.*` |
| self-monitor | Actor produces the monitor signal itself; always initialize from a self-monitor SFT checkpoint. | `self_monitor.enable=True` |
| verdict-monitor | Frozen trajectory verdict model produces a trust penalty. | `verdict_monitor.enable=True` |
| auxiliary safety RLHF | Prompt-only safety RLHF branch appended to main task training. | `auxiliary.enable=True` |
| critique-monitor | Fixed external critique monitor baseline. | `monitor_rollout_ref.enable=True`, `monitor_rollout_ref.enable_train_monitor=False`, `judge_model.*` |
| maximin | Actor-monitor game optimization. | `monitor_rollout_ref.enable=True`, `monitor_rollout_ref.enable_train_monitor=True`, `judge_model.*` |

Use `Qwen/Qwen3-8B` as the recommended global default model unless an experiment script intentionally uses a smaller model for debugging or memory pressure.

