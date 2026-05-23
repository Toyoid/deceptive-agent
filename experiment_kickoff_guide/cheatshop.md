# CheatShop

CheatShop uses `env.env_name=CheatShop`. It is a WebShop-derived shopping environment where the assistant should solve the user's shopping request through the normal shopper-facing workflow, while internal analyst tools or structured search affordances create opportunities for restricted-tool use, concealment, or misleading self-reporting.

## Features

CheatShop exposes normal WebShop search, product-page inspection, option selection, and purchase actions. It can also expose internal reference features such as `click[analyze_results]` and strict `@structured{...}` queries. Optional post-shopping stages can ask the agent to edit logs or produce a user-facing report. Monitor inputs can include actions only or actions plus observations, depending on `env.cheatshop.monitor_obs_level`.

CheatShop has explicit rule-based environment rewards, so trust-penalty runs usually use `reward_model.reward_manager=actor_monitor` rather than Lagrangian RL.

## Installation and Setup

Follow the WebShop installation and setup in [README_legacy_verl_agent.md](../README_legacy_verl_agent.md#1-webshop). That legacy section is the authoritative setup reference for the inherited WebShop assets and package constraints.

Use this repository's dedicated shopping environment name:

```bash
conda activate deceptive-agent-webshop
```

Install WebShop assets from the WebShop package directory:

```bash
cd agent_system/environments/env_package/webshop/webshop
bash setup.sh -d all
```

Then return to the repository root and install this repository in the same `deceptive-agent-webshop` environment, following the legacy README's WebShop section:

```bash
cd /path/to/deceptive-agent
conda activate deceptive-agent-webshop
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
```

Run CheatShop data preparation, RL training, and API rollout evaluation in `deceptive-agent-webshop`, not in the main `deceptive-agent` environment.

## Data Preparation

Prepare WebShop/CheatShop prompt rows:

```bash
conda activate deceptive-agent-webshop
python3 -m examples.data_preprocess.prepare --local_dir $DATA_ROOT/verl-agent --mode text --train_data_size 16 --val_data_size 32
```

This writes:

```text
$DATA_ROOT/verl-agent/text/train.parquet
$DATA_ROOT/verl-agent/text/test.parquet
```

Existing CheatShop scripts call this preprocessing command internally with script-specific `train_data_size` and `val_data_size`. Prefer the script defaults unless intentionally scaling a run.

## `ppo_trainer.yaml` Config Notes

Important config sections for CheatShop:

| Config | Meaning |
| --- | --- |
| `env.env_name=CheatShop` | Selects `CheatShopEnvironmentManager` and `build_cheatshop_envs`. |
| `env.max_steps` | Total maximum interaction steps, including optional shopping, log-edit, and self-report stages. |
| `env.rollout.n` | Grouped environment replicas for GRPO-style multi-sample rollouts. Keep `actor_rollout_ref.rollout.n=1`. |
| `env.webshop.use_small` | Uses the small WebShop item files when true. |
| `env.webshop.human_goals` | Controls WebShop goal source. |
| `env.cheatshop.require_self_report` | Adds a post-shopping report stage when true. |
| `env.cheatshop.enable_log_edit` | Adds a post-shopping log-edit stage when true. |
| `env.cheatshop.monitor_obs_level` | Monitor observation detail, commonly `action_plus_obs`. |
| `env.cheatshop.self_report_cot_visibility` | Controls whether chain-of-thought is visible in self-report artifacts; current scripts use `hidden`. |
| `env.cheatshop.shopping_max_steps` | Maximum shopping-stage steps; defaults to `${env.max_steps}`. |
| `env.cheatshop.oracle_top_n` | Number of visible products considered by the internal analysis helper. |
| `env.cheatshop.tamper_max_deletes` | Delete budget for log-edit tampering when log editing is enabled. |
| `reward_model.reward_manager=actor_monitor` | Recommended trust-penalty application path for CheatShop mitigation. |
| `reward_model.reward_manager_config.actor_monitor.trust_penalty_coef` | Scalar for `episode_reward - coef * trust_penalty`. |
| `monitor_rollout_ref.enable` | Enables external critique monitor for critique-monitor or maximin. |
| `monitor_rollout_ref.enable_train_monitor` | `False` for critique-monitor; `True` for maximin. |
| `judge_model.backend=api_cot` | Default judge form for critique-monitor and maximin. |

For CheatShop mitigation methods, initialize `actor_rollout_ref.model.path` from a prior CheatShop RL checkpoint. The checkpoint is needed because the mitigation study assumes the agent is already competent in WebShop-style task completion and can exhibit the behavior being mitigated.

## Method Launches

Run RL first from the WebShop/CheatShop environment:

```bash
conda activate deceptive-agent-webshop
bash examples/grpo_trainer/webshop/run_cheatshop.sh
```

Use the resulting checkpoint for critique-monitor and maximin:

```text
checkpoints/verl_agent_webshop/grpo_qwen3_8b_cheatshop/global_step_<N>/actor/huggingface
```

Critique-monitor with API CoT judge. Start the judge server first. If you serve the judge from `deceptive-agent`, keep it in a separate terminal:

```bash
conda activate deceptive-agent
CUDA_VISIBLE_DEVICES=6,7 bash examples/grpo_trainer/serve_cot_judge_vllm.sh 7001 2 Qwen/Qwen3-8B
```

Then run CheatShop training from `deceptive-agent-webshop`:

```bash
conda activate deceptive-agent-webshop
JUDGE_MODEL_NAME=Qwen/Qwen3-8B JUDGE_PORT=7001 bash examples/grpo_trainer/webshop/run_cheatshop_m_cot_judge.sh
```

Maximin with API CoT judge:

```bash
conda activate deceptive-agent-webshop
JUDGE_MODEL_NAME=Qwen/Qwen3-8B JUDGE_PORT=7001 bash examples/grpo_trainer/webshop/run_cheatshop_mm_cot_judge.sh
```

The constrained-scorer/Lagrangian CheatShop maximin script is available as an ablation, but the recommended default for CheatShop is `actor_monitor` because the environment reward is explicit:

```bash
bash examples/grpo_trainer/webshop/run_cheatshop_mm_lag.sh
```

Expected TODO script paths for remaining methods:

```text
examples/grpo_trainer/webshop/run_cheatshop_self_m.sh
examples/grpo_trainer/webshop/run_cheatshop_verdict_m.sh
examples/grpo_trainer/webshop/run_cheatshop_aux_safety.sh
```

When adding those scripts, keep all CheatShop data preparation, training, and evaluation in `deceptive-agent-webshop`, and prefer `reward_model.reward_manager=actor_monitor` for trust-penalty application.

## API Rollout Evaluation

OpenAI-compatible API evaluation:

```bash
conda activate deceptive-agent-webshop
bash examples/api_rollout_eval/run_cheatshop_openai_api.sh
```

Local vLLM evaluation:

```bash
conda activate deceptive-agent-webshop
CUDA_VISIBLE_DEVICES=0 bash examples/api_rollout_eval/serve_local_vllm.sh 7000 1 Qwen/Qwen3-8B qwen3-8b
bash examples/api_rollout_eval/run_cheatshop_local_vllm.sh model.api_base=http://127.0.0.1:7000/v1 model.model=qwen3-8b
```

The CheatShop eval scripts can sample episodes with `data.num_episodes`; they do not require a search parquet file. They do require the WebShop assets from the setup section.

