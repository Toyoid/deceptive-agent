# SearchQA

SearchQA trains a multi-turn QA agent with public retrieval. The standard task uses `env.env_name=search`; the unauthorized-tool setting uses `deceptive_search`, which additionally exposes an insider answer lookup and records use, disclosure, citation, and audit outcomes.

## Prepare

Install the Search package, retriever environment, index, and corpus as described in the root [SearchQA installation guide](../README.md#3-searchqa).
Then before training or evaluation, start retrieval in its dependency environment:

```bash
conda activate retriever
DATA_ROOT="$DATA_ROOT" bash examples/search/retriever/retrieval_launch.sh
```

The default endpoint is `http://127.0.0.1:8000/retrieve`.

## Launch

| Setting / method | Primary script |
| --- | --- |
| SearchQA without unauthorized tools | `examples/grpo_trainer/search_qa/run_search.sh` |
| RL, with unauthorized tools | `examples/grpo_trainer/search_qa/run_deceptive_search.sh` |
| Mixed Safety RLHF | `examples/grpo_trainer/search_qa/run_deceptive_search_aux_safety.sh` |
| Self-Monitor | `examples/grpo_trainer/search_qa/run_deceptive_search_self_m.sh` |
| Verdict Monitor | `examples/grpo_trainer/search_qa/run_deceptive_search_verdict_m.sh` |
| Critique Monitor | `examples/grpo_trainer/search_qa/run_deceptive_search_m_cot_judge.sh` |
| e-PoG | `examples/grpo_trainer/search_qa/run_deceptive_search_mm_cot_judge.sh` |

Keep the retriever running and launch with `bash <script>`. Mitigation scripts contain explicit actor/monitor checkpoint paths; update them rather than assuming the checked-in cluster paths exist.

## Evaluate

With the retriever running:

```bash
bash examples/api_rollout_eval/run_deceptive_search_openai_api.sh
```

For a local checkpoint, use `serve_local_vllm.sh` followed by `run_deceptive_search_local_vllm.sh`. The evaluator expects the processed test parquet and accepts overrides such as `model.api_base` and `model.model`.
