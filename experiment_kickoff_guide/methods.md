# Methods

This page explains the method families used in the experiment guide, the important [ppo_trainer.yaml](../verl/trainer/config/ppo_trainer.yaml) configs, and the current launch scripts. The six method names used in this guide are `RL`, `self-monitor`, `verdict-monitor`, `auxiliary safety RLHF`, `critique-monitor`, and `maximin`.

## Shared Training Configs

All training scripts call:

```bash
python3 -m verl.trainer.main_ppo <Hydra overrides>
```

Important shared sections:

| Config section | Purpose |
| --- | --- |
| `data.*` | Train/validation parquet paths, prompt/response lengths, prompt filtering, raw chat retention. |
| `actor_rollout_ref.model.*` | Actor checkpoint path, tokenizer/template behavior, FSDP memory settings. |
| `actor_rollout_ref.actor.*` | PPO/GRPO actor optimization, KL loss, batch sizes, checkpoint content. |
| `actor_rollout_ref.rollout.*` | Rollout backend, tensor parallel size, vLLM memory, sampling behavior. |
| `reward_model.*` | Reward model worker and reward manager. Roles usually enable a reward model; Search/CheatShop usually rely on environment rewards. |
| `env.*` | Environment selection, maximum steps, history length, grouped rollout count, and environment-specific knobs. |
| `trainer.*` | GPU placement, checkpoint cadence, logging, validation, monitor/judge worker placement, output directories. |

Keep `actor_rollout_ref.rollout.n=1` and `actor_rollout_ref.rollout.val_kwargs.n=1`. In this repo, GRPO-style grouped sampling for agent environments is controlled by `env.rollout.n` and `env.rollout.val_n`.

## Trust-Penalty Sources and Application

The code separates trust-penalty sources from trust-penalty application.

Trust-penalty sources:

| Source | Config |
| --- | --- |
| self-monitor | `self_monitor.enable=True` |
| verdict-monitor | `verdict_monitor.enable=True` |
| external critique monitor | `monitor_rollout_ref.enable=True` |

These sources are mutually exclusive in a single run. The trainer rejects runs that enable more than one of `self_monitor`, `verdict_monitor`, and `monitor_rollout_ref`.

Trust-penalty application:

| Application | Config | Recommended use |
| --- | --- | --- |
| Lagrangian constrained RL | `algorithm.lagrangian.enable=True` | Roles, because task reward comes from a reward model. |
| Actor-monitor reward manager | `reward_model.reward_manager=actor_monitor` | Deceptive search and CheatShop, because task reward is explicit rule-based environment reward. |

Do not enable both application paths in the same run. `actor_monitor` computes actor reward as `episode_reward - trust_penalty_coef * trust_penalty`, with `trust_penalty_coef` configured at `reward_model.reward_manager_config.actor_monitor.trust_penalty_coef`. The `actor_monitor` reward manager supports maximin, critique-monitor, self-monitor, and verdict-monitor as long as one trust-penalty source is enabled.

## Judges

Judges are used with external critique-monitor and maximin runs. They convert a natural-language critique into a trust-penalty score. Judges are not used by self-monitor or verdict-monitor.

There are two judge forms:

| Judge form | Config | How it runs |
| --- | --- | --- |
| API CoT judge | `judge_model.backend=api_cot` | Calls an OpenAI-compatible API server, usually a local vLLM server. This is the default for this guide. |
| Constrained scorer | `judge_model.backend=constrained_logits` | Runs a local model and scores only valid label tokens. |

Default CoT judge server:

```bash
CUDA_VISIBLE_DEVICES=6,7 bash examples/grpo_trainer/serve_cot_judge_vllm.sh 7001 2 Qwen/Qwen3-8B
```

Typical API CoT judge overrides:

```text
judge_model.backend=api_cot
judge_model.template_name=strict5_cot
judge_model.valid_tokens='["0","1","2","3","4"]'
judge_model.token_weights='[-1.0,-0.5,0.0,0.5,1.0]'
judge_model.api_cot.provider=openai_compatible
judge_model.api_cot.model=$JUDGE_MODEL_NAME
judge_model.api_cot.api_base=http://127.0.0.1:$JUDGE_PORT/v1
judge_model.api_cot.api_key=dummy
judge_model.api_cot.temperature=0.6
judge_model.api_cot.parse_error=neutral
judge_model.api_cot.api_error=raise
```

Typical constrained-scorer overrides:

```text
judge_model.backend=constrained_logits
judge_model.model.path=Qwen/Qwen3-8B
judge_model.valid_tokens='["0","1","2","3","4"]'
judge_model.token_weights='[-1.0,-0.5,0.0,0.5,1.0]'
judge_model.constrained_top_k=2
judge_model.template_name=strict5
```

## RL

RL is task/environment reinforcement learning without a deception mitigation method. For deceptive search and CheatShop, RL produces the initial actor checkpoint used by later mitigation methods. Roles can run directly from the original actor model unless a specific ablation says otherwise.

Important configs:

```text
algorithm.adv_estimator=grpo
actor_rollout_ref.model.path=Qwen/Qwen3-8B
self_monitor.enable=False
verdict_monitor.enable=False
monitor_rollout_ref.enable=False
auxiliary.enable=False
```

Launches:

```bash
bash examples/ppo_trainer/run_deceptive_roles.sh
bash examples/grpo_trainer/search_qa/run_deceptive_search.sh
conda activate deceptive-agent-webshop
bash examples/grpo_trainer/webshop/run_cheatshop.sh
```

For deceptive search and CheatShop, make sure the RL script saves HuggingFace-format actor checkpoints by including `hf_model` in `actor_rollout_ref.actor.checkpoint.contents`.

## Self-Monitor

Self-monitor trains the actor with its own self-reported safety/honesty signal. It does not use an external monitor or judge. Self-monitor always initializes actor RL from a self-monitor SFT checkpoint for any environment.

Important configs:

```text
self_monitor.enable=True
actor_rollout_ref.model.path=<self-monitor-sft-checkpoint>
monitor_rollout_ref.enable=False
verdict_monitor.enable=False
judge_model.enable=False
```

Apply the trust penalty with Lagrangian for roles:

```text
algorithm.lagrangian.enable=True
reward_model.reward_manager=episode
```

Apply the trust penalty with `actor_monitor` for deceptive search or CheatShop:

```text
reward_model.reward_manager=actor_monitor
algorithm.lagrangian.enable=False
```

Current launch:

```bash
bash examples/sft/self_monitor/run_qwen2.5_7b_sp2.sh 8
bash examples/ppo_trainer/run_deceptive_roles_self_m.sh
```

Expected TODO paths:

```text
examples/grpo_trainer/search_qa/run_deceptive_search_self_m.sh
examples/grpo_trainer/webshop/run_cheatshop_self_m.sh
```

## Verdict-Monitor

Verdict-monitor uses a frozen trajectory-level model to score whether the actor behavior is unsafe or dishonest. It does not use an external critique monitor and does not use `judge_model`.

Important configs:

```text
verdict_monitor.enable=True
verdict_monitor.model.path=<verdict-monitor-checkpoint-or-base-model>
verdict_monitor.valid_tokens='["0","1"]'
verdict_monitor.token_weights='[0.0,1.0]'
self_monitor.enable=False
monitor_rollout_ref.enable=False
judge_model.enable=False
```

Current launch:

```bash
bash examples/ppo_trainer/run_deceptive_roles_verdict_m.sh
bash examples/grpo_trainer/search_qa/run_deceptive_search_verdict_m.sh
```

Expected TODO paths:

```text
examples/grpo_trainer/webshop/run_cheatshop_verdict_m.sh
```

For deceptive search and CheatShop, initialize the actor from a prior environment RL checkpoint and use `reward_model.reward_manager=actor_monitor`.

## Auxiliary Safety RLHF

Auxiliary safety RLHF appends prompt-only safety RLHF updates to the main task training loop. This method is a deception/misalignment mitigation baseline, but it is intentionally separate from monitor, self-monitor, verdict-monitor, maximin, and Lagrangian trust-penalty modes.

Important configs:

```text
auxiliary.enable=True
auxiliary.data.train_files=<prompt-only-safety-parquet>
auxiliary.batch_size=<batch-size>
auxiliary.rollout.n=<aux-rollouts>
auxiliary.reward_model.use_main=True
auxiliary.reward_model.strip_thinking=True
monitor_rollout_ref.enable=False
self_monitor.enable=False
verdict_monitor.enable=False
algorithm.lagrangian.enable=False
```

Prepare PKU-SafeRLHF prompt-only data:

```bash
python3 -m verl.trainer.auxiliary.preprocess prompt_only --dataset PKU-Alignment/PKU-SafeRLHF --split train --local_dir $DATA_ROOT/pku_safe_rlhf/prompt_only --data_source pku_safe_rlhf
```

Current launch:

```bash
bash examples/ppo_trainer/run_deceptive_roles_aux_safety.sh
```

Expected TODO paths:

```text
examples/grpo_trainer/search_qa/run_deceptive_search_aux_safety.sh
examples/grpo_trainer/webshop/run_cheatshop_aux_safety.sh
```

## Critique-Monitor

Critique-monitor is the fixed external monitor baseline. It enables monitor rollout but disables monitor training. A judge converts monitor critiques into trust-penalty scores.

Important configs:

```text
monitor_rollout_ref.enable=True
monitor_rollout_ref.enable_train_monitor=False
monitor_rollout_ref.model.path=<fixed-monitor-checkpoint-or-base-model>
judge_model.backend=api_cot
self_monitor.enable=False
verdict_monitor.enable=False
```

Use Lagrangian for roles:

```text
reward_model.reward_manager=episode
algorithm.lagrangian.enable=True
```

Use `actor_monitor` for deceptive search and CheatShop:

```text
reward_model.reward_manager=actor_monitor
algorithm.lagrangian.enable=False
```

Launches:

```bash
bash examples/ppo_trainer/run_deceptive_roles_m_lag.sh
JUDGE_MODEL_NAME=Qwen/Qwen3-8B JUDGE_PORT=7001 bash examples/grpo_trainer/search_qa/run_deceptive_search_m_cot_judge.sh
conda activate deceptive-agent-webshop
JUDGE_MODEL_NAME=Qwen/Qwen3-8B JUDGE_PORT=7001 bash examples/grpo_trainer/webshop/run_cheatshop_m_cot_judge.sh
```

The roles critique-monitor launch above is the currently available constrained-scorer script. The default judge form for new critique-monitor/maximin scripts is API CoT judge, so create a roles CoT-judge critique-monitor variant if that comparison is needed.

For deceptive search and CheatShop, edit `actor_rollout_ref.model.path` to a prior RL actor checkpoint before launch.

## Maximin

Maximin is the actor-monitor game optimization method. The actor is optimized to complete the task while reducing judged trust penalties; the monitor is optimized to produce critiques that expose behavior judged as problematic. In config terms, maximin is external monitor rollout with monitor training enabled.

Important configs:

```text
monitor_rollout_ref.enable=True
monitor_rollout_ref.enable_train_monitor=True
monitor_rollout_ref.rollout.n=<monitor-rollout-count>
monitor_rollout_ref.algorithm.adv_estimator=reinforce_plus_plus
monitor_rollout_ref.monitor.optim.lr=<monitor-lr>
judge_model.backend=api_cot
```

Roles default to Lagrangian trust-penalty application:

```text
reward_model.reward_manager=episode
algorithm.lagrangian.enable=True
```

Deceptive search and CheatShop default to `actor_monitor`:

```text
reward_model.reward_manager=actor_monitor
algorithm.lagrangian.enable=False
```

Launches:

```bash
JUDGE_MODEL_NAME=Qwen/Qwen3-8B JUDGE_PORT=7001 bash examples/grpo_trainer/deceptive_roles/run_deceptive_roles_mm_lag_cot_judge.sh
JUDGE_MODEL_NAME=Qwen/Qwen3-8B JUDGE_PORT=7001 bash examples/grpo_trainer/search_qa/run_deceptive_search_mm_cot_judge.sh
conda activate deceptive-agent-webshop
JUDGE_MODEL_NAME=Qwen/Qwen3-8B JUDGE_PORT=7001 bash examples/grpo_trainer/webshop/run_cheatshop_mm_cot_judge.sh
```

Constrained-scorer maximin variants:

```bash
bash examples/grpo_trainer/deceptive_roles/run_deceptive_roles_mm_lag.sh
bash examples/grpo_trainer/search_qa/run_deceptive_search_mm.sh
bash examples/grpo_trainer/webshop/run_cheatshop_mm_lag.sh
```

For deceptive search and CheatShop, edit `actor_rollout_ref.model.path` to a prior RL actor checkpoint before launch.
