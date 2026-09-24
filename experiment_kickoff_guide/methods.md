# Methods

All paper methods share the same actor/environment rollout stack. They differ in where the safety signal comes from and whether the monitor is fixed or strategically trained.

| Paper name | Purpose | Key switch / script marker |
| --- | --- | --- |
| RL | Optimize task reward without a monitoring intervention | base script |
| Mixed Safety RLHF | Mix task RL with prompt-only safety training | `auxiliary.enable=True`, `_aux_safety` |
| Self-Monitor | The actor emits its own assessment; no external monitor model | `self_monitor.enable=True`, `_self_m` |
| Verdict Monitor | A frozen model directly scores the trajectory | `verdict_monitor.enable=True`, `_verdict_m` |
| Critique Monitor | A frozen monitor writes a critique that a judge scores | `monitor_rollout_ref.enable=True`, `enable_train_monitor=False`, `_m_` |
| e-PoG | Joint agent–monitor game: the monitor is trained against the actor | `monitor_rollout_ref.enable=True`, `enable_train_monitor=True`, `_mm_` |

The code may call e-PoG “maximin” in comments and configuration. We recommend using **e-PoG** in new user-facing documentation and experiment labels.

## Choose the reward path

Agentic Role-Playing obtains task utility from a reward model. Use `reward_model.reward_manager=episode` with `algorithm.lagrangian.enable=True` to incorporate penalty signals from monitors.

SearchQA and WebShop have explicit environment rewards. Their monitored methods normally use `reward_model.reward_manager=actor_monitor`, which combines episode reward with the selected trust penalty. Do not combine this with the Lagrangian path.

Only one trust source—Self-Monitor, Verdict Monitor, or external Critique Monitor/e-PoG—should be enabled in a run.

## Self-Monitor prerequisite

Train or obtain a Self-Monitor SFT actor before Self-Monitor RL:

```bash
bash examples/sft/self_monitor/run_qwen3_8b.sh 8
# also available: run_qwen3_4b.sh, run_gemma3_4b.sh
```

Set `self_monitor_sft_ckpt` in the selected RL script to the checkpoint after SFT.

## Critique Monitor and e-PoG prerequisite

The primary scripts use an OpenAI-compatible chain-of-thought judge. Start it on separate GPUs:

```bash
CUDA_VISIBLE_DEVICES=6,7 \
  bash examples/grpo_trainer/vllm_serve_cot_judge.sh \
  7001 2 hahnli/Qwen3-8B-CoT-Judge Qwen3-8B-GRM
```

Then set `JUDGE_PORT` / `JUDGE_MODEL_NAME` or the corresponding `judge_model.api_cot.*` overrides in the training script. The judge maps a natural-language critique to a scalar trust penalty. Some older/ablation configurations use the in-process `constrained_logits` judge instead.

## Reading a launch script

The useful control surface is small:

- `data.*`: parquet paths and sequence limits.
- `actor_rollout_ref.*`: actor checkpoint, optimization, rollout engine, and parallelism.
- `env.*`: environment, step budget, grouped samples, and unauthorized-tool options.
- `reward_model.*`: task reward and trust-penalty composition.
- `self_monitor`, `verdict_monitor`, `monitor_rollout_ref`, `judge_model`: oversight components.
- `trainer.*`: GPU placement, logging, validation, checkpointing, and rollout dumps.

For agent environments, set grouped sampling with `env.rollout.n` and keep `actor_rollout_ref.rollout.n=1`.
