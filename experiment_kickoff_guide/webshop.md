# WebShop

WebShop trains an agent to search, inspect, configure, and buy products. The standard setting uses `env.env_name=Webshop`; the unauthorized-tool setting uses `CheatShop`, which adds privileged analysis/search affordances and monitor-visible traces. Optional self-report and log-edit stages support research on disclosure and tampering.

## Prepare

Activate the dedicated environment and install the simulator assets by following the root [WebShop installation guide](../README.md#2-webshop).
Then create prompt rows once; most current scripts repeat this step with their own sizes:

```bash
python3 -m examples.data_preprocess.prepare \
  --local_dir "$DATA_ROOT/verl-agent" --mode text \
  --train_data_size 16 --val_data_size 128
```

## Launch

| Setting / method | Primary script |
| --- | --- |
| WebShop without unauthorized tools | `examples/grpo_trainer/webshop/run_webshop.sh` |
| RL, with unauthorized tools | `examples/grpo_trainer/webshop/run_cheatshop.sh` |
| Mixed Safety RLHF | `examples/grpo_trainer/webshop/run_cheatshop_aux_safety.sh` |
| Self-Monitor | `examples/grpo_trainer/webshop/run_cheatshop_self_m.sh` |
| Verdict Monitor | `examples/grpo_trainer/webshop/run_cheatshop_verdict_m.sh` |
| Critique Monitor | `examples/grpo_trainer/webshop/run_cheatshop_m_cot_judge.sh` |
| e-PoG | `examples/grpo_trainer/webshop/run_cheatshop_mm_cot_judge.sh` |

Launch with `bash <script>` from the repository root while `deceptive-agent-webshop` is active. Model-specific Qwen3-4B and Gemma-3-4B scripts are colocated. Update the checked-in data and checkpoint paths first.

The standard task also has PPO, RLOO, DAPO, and GiGPO examples under `examples/*_trainer/`; these demonstrate the framework's breadth, while the table above is the paper-aligned experiment path.

## Evaluate

```bash
bash examples/api_rollout_eval/run_cheatshop_openai_api.sh
```

For a local checkpoint, run `examples/api_rollout_eval/serve_local_vllm.sh` and then `run_cheatshop_local_vllm.sh`. Evaluation must use the WebShop environment because it loads the installed catalog and simulator assets.
