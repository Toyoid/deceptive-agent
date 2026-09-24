# Agentic Role-Playing

Agentic Role-Playing tests whether an assistant follows a deceptive role or the user-aligned objective. In code it is the single-turn `ReasonChat` environment. The environment itself returns no task reward; a reward model scores the answer, so monitor penalties use constrained (Lagrangian) RL.

## Prepare

Install and activate the main `deceptive-agent` environment as described in the root [Installation guide](../README.md#1-agentic-role-playing). Create the parquet files once (many launch scripts also do this):

```bash
python3 examples/data_preprocess/deceptive_roles.py \
  --local_dir "$DATA_ROOT/deceptive_roles"
```

The source data is under `agent_system/environments/env_package/reason_chat/deceptive_roles`. Rows retain role metadata used by the environment and monitors.

## Launch

| Method | Primary script |
| --- | --- |
| RL | `examples/grpo_trainer/deceptive_roles/run_deceptive_roles.sh` |
| Mixed Safety RLHF | `examples/grpo_trainer/deceptive_roles/run_deceptive_roles_aux_safety.sh` |
| Self-Monitor | `examples/grpo_trainer/deceptive_roles/run_deceptive_roles_self_m.sh` |
| Verdict Monitor | `examples/grpo_trainer/deceptive_roles/run_deceptive_roles_verdict_m.sh` |
| Critique Monitor | `examples/grpo_trainer/deceptive_roles/run_deceptive_roles_m_lag_cot_judge.sh` |
| e-PoG | `examples/grpo_trainer/deceptive_roles/run_deceptive_roles_mm_lag_cot_judge.sh` |

Launch a row with `bash <script>`. Model-specific Qwen3-4B and Gemma-3-4B variants live beside these scripts. The `*_w2s.sh` e-PoG script is the weak-to-strong oversight variant.

Self-Monitor requires a Self-Monitor SFT actor; see [methods.md](methods.md). Critique Monitor and e-PoG require the CoT judge service. Confirm the actor, monitor, reward-model, and data paths inside the chosen script before running.

## Evaluate

```bash
bash examples/api_rollout_eval/run_deceptive_roles_openai_api.sh
```

For a local checkpoint, first serve it with `examples/api_rollout_eval/serve_local_vllm.sh`, then use `run_deceptive_roles_local_vllm.sh`.
