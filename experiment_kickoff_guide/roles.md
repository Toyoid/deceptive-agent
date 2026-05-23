# Roles / ReasonChat

Roles experiments use `env.env_name=ReasonChat`. The environment is a lightweight single-turn chat task built around deceptive role prompts under `agent_system/environments/env_package/reason_chat`. The environment returns zero task reward by itself; the task reward is supplied by a reward model, which is why roles experiments usually apply trust penalties through Lagrangian constrained RL instead of `reward_model.reward_manager=actor_monitor`.

## Features

ReasonChat loads per-sample prompt metadata from the preprocessed parquet rows. Each rollout must produce a `<think>...</think>` block and then an answer, preferably in `<answer>...</answer>` tags. `reason_chat_projection` extracts the reasoning and answer fields, validates the tag structure, and passes the final assistant response into environment metadata. Monitor and judge prompts can then use `monitor_background`, `agent_trajectory`, `user_input`, `evidence`, and `agent_response` from the episode info.

The current environment is single-turn. Set `env.max_steps=1`; a multi-turn ReasonChat variant would need additional environment responses after each step.

## Installation and Setup

Roles uses the main training environment:

```bash
conda activate deceptive-agent
cd /path/to/deceptive-agent
```

No separate environment service is required. The root [README.md](../README.md) contains the main training stack installation. Unlike deceptive search and cheatshop, roles does not require Search-R1 retriever setup or WebShop setup.

## Data Preparation

Default deceptive-roles data:

```bash
python3 examples/data_preprocess/deceptive_roles.py --local_dir $DATA_ROOT/deceptive_roles
```

This writes:

```text
$DATA_ROOT/deceptive_roles/train.parquet
$DATA_ROOT/deceptive_roles/test.parquet
```

Some CoT-judge maximin scripts use the improved role set and disable prompt formatting in preprocessing:

```bash
python3 examples/data_preprocess/deceptive_roles.py --local_dir $DATA_ROOT/deceptive_roles_improved --source_dir agent_system/environments/env_package/reason_chat/deceptive_roles_improved --no_format_prompt
```

When reproducing a specific run, prefer the preprocessing command embedded in that script.

## `ppo_trainer.yaml` Config Notes

Important config sections for roles:

| Config | Meaning |
| --- | --- |
| `env.env_name=ReasonChat` | Selects `ReasonChatEnvironmentManager` and `build_reason_chat_envs`. |
| `env.max_steps=1` | Keeps the role task single-turn. |
| `env.rollout.n` and `env.rollout.val_n` | Grouped environment replicas used for GRPO-style multi-sample rollouts. Keep `actor_rollout_ref.rollout.n=1`; this repo implements grouping through `env.rollout.n`. |
| `data.return_raw_chat=True` | Keeps chat-format fields needed by reward model, monitor, and judge paths. |
| `reward_model.enable=True` | Enables the reward model that supplies the task reward for roles. |
| `reward_model.model.path` | Reward model checkpoint, for example `sfairXC/FsfairX-LLaMA3-RM-v0.1`. |
| `reward_model.normalization.enable=True` | Normalizes reward model scores before RL updates. |
| `algorithm.lagrangian.enable=True` | Recommended trust-penalty application path for roles mitigation methods. |
| `algorithm.lagrangian.threshold` | Allowed average trust-penalty/cost target. |
| `algorithm.lagrangian.lambda_init`, `lambda_max`, `lambda_lr`, `lambda_update_delay_steps` | Lagrange multiplier initialization and update schedule. |
| `monitor_rollout_ref.enable` | Enables an external critique monitor for critique-monitor or maximin. |
| `monitor_rollout_ref.enable_train_monitor` | `False` for critique-monitor; `True` for maximin. |
| `self_monitor.enable` | Enables the actor-output self-monitor parser. Mutually exclusive with external monitor and verdict monitor. |
| `verdict_monitor.enable` | Enables frozen verdict-monitor scoring. Mutually exclusive with external monitor and self-monitor. |
| `judge_model.backend` | Used only with external critique-monitor/maximin. Use `api_cot` by default or `constrained_logits` for constrained scorer runs. |

The trainer enforces that `self_monitor`, `verdict_monitor`, and `monitor_rollout_ref` are mutually exclusive. It also enforces that a trust-penalty source must be paired with exactly one application path: either `algorithm.lagrangian.enable=True` or `reward_model.reward_manager=actor_monitor`.

## Method Launches

RL baseline:

```bash
bash examples/ppo_trainer/run_deceptive_roles.sh
```

Self-monitor, after setting `self_monitor_sft_ckpt` to a self-monitor SFT checkpoint:

```bash
bash examples/sft/self_monitor/run_qwen2.5_7b_sp2.sh 8
bash examples/ppo_trainer/run_deceptive_roles_self_m.sh
```

Verdict-monitor:

```bash
bash examples/ppo_trainer/run_deceptive_roles_verdict_m.sh
```

Auxiliary safety RLHF:

```bash
bash examples/ppo_trainer/run_deceptive_roles_aux_safety.sh
```

Critique-monitor with Lagrangian trust penalty:

```bash
bash examples/ppo_trainer/run_deceptive_roles_m_lag.sh
```

This existing roles critique-monitor script uses the constrained scorer. The default judge form for new critique-monitor/maximin scripts is API CoT judge, so add a CoT-judge critique-monitor script if you need the default judge setup for roles.

Maximin with API CoT judge. Start the judge server first:

```bash
CUDA_VISIBLE_DEVICES=6,7 bash examples/grpo_trainer/serve_cot_judge_vllm.sh 7001 2 Qwen/Qwen3-8B
```

Then run training:

```bash
JUDGE_MODEL_NAME=Qwen/Qwen3-8B JUDGE_PORT=7001 bash examples/grpo_trainer/deceptive_roles/run_deceptive_roles_mm_lag_cot_judge.sh
```

The constrained-scorer maximin variant is available at:

```bash
bash examples/grpo_trainer/deceptive_roles/run_deceptive_roles_mm_lag.sh
```

## API Rollout Evaluation

OpenAI-compatible API:

```bash
conda activate deceptive-agent
bash examples/api_rollout_eval/run_deceptive_roles_openai_api.sh
```

Local vLLM:

```bash
conda activate deceptive-agent
CUDA_VISIBLE_DEVICES=0 bash examples/api_rollout_eval/serve_local_vllm.sh 7000 1 Qwen/Qwen3-8B qwen3-8b
bash examples/api_rollout_eval/run_deceptive_roles_local_vllm.sh model.api_base=http://127.0.0.1:7000/v1 model.model=qwen3-8b
```

The roles eval script expects the roles parquet, usually `$DATA_ROOT/deceptive_roles/train.parquet`. Run the data-prep command above if it is missing.
