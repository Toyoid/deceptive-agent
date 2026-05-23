# Quick Start

This page gives the shortest concrete path to launch experiments after installation is complete. It does not include installation details; use [roles.md](roles.md), [deceptive_search.md](deceptive_search.md), and [cheatshop.md](cheatshop.md) for environment setup.

## Global Assumptions

Run commands from the repository root:

```bash
cd /path/to/deceptive-agent
```

Use an 8-GPU node. Most scripts are cluster templates and set their own `CUDA_VISIBLE_DEVICES`, GPU counts, model paths, and data roots. Before launching, edit the target script or override these values:

```text
DATA_ROOT
actor_rollout_ref.model.path
monitor_rollout_ref.model.path
verdict_monitor.model.path
judge_model.model.path
reward_model.model.path
trainer.project_name
trainer.experiment_name
trainer.n_gpus_per_node
trainer.n_gpus_per_node_monitor
trainer.judge_pool_mode
trainer.rollout_data_dir
```

Use `Qwen/Qwen3-8B` as the recommended default actor and judge model unless a script intentionally uses a smaller checkpoint for debugging.

## Methods Overview

| Method | Launch group in this guide | Initialization | Extra requirements |
| --- | --- | --- | --- |
| RL | `RL, Verdict-Monitor, and Auxiliary Safety RLHF` | Original actor model, usually `Qwen/Qwen3-8B`. | For deceptive search and CheatShop, this produces the prior RL actor checkpoint used by later mitigation methods. |
| verdict-monitor | `RL, Verdict-Monitor, and Auxiliary Safety RLHF` | Roles can start from the original actor model; deceptive search and CheatShop should start from a prior environment RL checkpoint. | Set `verdict_monitor.model.path`; no external critique judge service. |
| auxiliary safety RLHF | `RL, Verdict-Monitor, and Auxiliary Safety RLHF` | Roles can start from the original actor model; deceptive search and CheatShop should start from a prior environment RL checkpoint. | Prepare prompt-only auxiliary safety data if the target script does not do it internally. |
| self-monitor | `Self-Monitor` | Self-monitor SFT checkpoint. | Train or locate the SFT checkpoint first, then set the RL script's actor path to it. |
| critique-monitor | `Critique-Monitor and Maximin` | Roles use the script default; deceptive search and CheatShop should start from a prior environment RL checkpoint. | External critique monitor plus judge; default judge is API CoT and needs a judge service. |
| maximin | `Critique-Monitor and Maximin` | Roles use the script default; deceptive search and CheatShop should start from a prior environment RL checkpoint. | Trainable critique monitor plus judge; default judge is API CoT and needs a judge service. |

## Conda Environments

| Environment | Training conda env | Extra service env | Notes |
| --- | --- | --- | --- |
| roles | `deceptive-agent` | none | Main training stack only. |
| deceptive search | `deceptive-agent` | `retriever` | Training runs in `deceptive-agent`; the Search-R1 retriever runs in `retriever`. |
| cheatshop | `deceptive-agent-webshop` | none | Data prep, training, and API rollout evaluation run in the WebShop/CheatShop env. |

## Roles

### Shared Prep

Use the main training environment:

```bash
conda activate deceptive-agent
```

Prepare roles data only if the target script does not already do it, or if you need the parquet for API rollout evaluation:

```bash
python3 examples/data_preprocess/deceptive_roles.py --local_dir $DATA_ROOT/deceptive_roles
```

Most checked-in roles training scripts already call their required data-preparation command. The maximin CoT script uses the improved roles source and `--no_format_prompt` internally, so follow the script when reproducing that run.

### RL, Verdict-Monitor, and Auxiliary Safety RLHF

#### RL

```bash
bash examples/ppo_trainer/run_deceptive_roles.sh
```

#### Verdict-Monitor

```bash
bash examples/ppo_trainer/run_deceptive_roles_verdict_m.sh
```

#### Auxiliary Safety RLHF

```bash
bash examples/ppo_trainer/run_deceptive_roles_aux_safety.sh
```

The roles auxiliary script currently prepares PKU-SafeRLHF prompt-only data internally. If you write a custom script that does not, prepare it with:

```bash
python3 -m verl.trainer.auxiliary.preprocess prompt_only --dataset PKU-Alignment/PKU-SafeRLHF --split train --local_dir $DATA_ROOT/pku_safe_rlhf/prompt_only --data_source pku_safe_rlhf
```

### Self-Monitor

#### Self-Monitor SFT

If no self-monitor SFT checkpoint exists yet, train one first:

```bash
bash examples/sft/self_monitor/run_qwen2.5_7b_sp2.sh 8
```

#### Self-Monitor RL

Set `self_monitor_sft_ckpt` in the training script to the SFT checkpoint, then launch:

```bash
bash examples/ppo_trainer/run_deceptive_roles_self_m.sh
```

### Critique-Monitor and Maximin

These methods use an external critique monitor and a judge. Use `judge_model.backend=api_cot` by default; start a CoT judge service before API-CoT runs. If using `judge_model.backend=constrained_logits`, no API judge service is needed, but the local judge worker still needs GPU placement through `trainer.judge_pool_mode`, `trainer.n_gpus_per_node_judge`, and related script settings.

#### CoT Judge Service

```bash
conda activate deceptive-agent
CUDA_VISIBLE_DEVICES=6,7 bash examples/grpo_trainer/serve_cot_judge_vllm.sh 7001 2 Qwen/Qwen3-8B
```

#### Critique-Monitor

Current constrained-scorer script:

```bash
bash examples/ppo_trainer/run_deceptive_roles_m_lag.sh
```

The default judge form for new critique-monitor/maximin scripts is API CoT judge. The checked-in roles critique-monitor script above is the existing constrained-scorer variant; add a CoT-judge critique-monitor script if that comparison is needed.

#### Maximin

API CoT judge script:

```bash
JUDGE_MODEL_NAME=Qwen/Qwen3-8B JUDGE_PORT=7001 bash examples/grpo_trainer/deceptive_roles/run_deceptive_roles_mm_lag_cot_judge.sh
```

Constrained-scorer variant:

```bash
bash examples/grpo_trainer/deceptive_roles/run_deceptive_roles_mm_lag.sh
```

## Deceptive Search

### Shared Prep

Use `deceptive-agent` for data preparation and training:

```bash
conda activate deceptive-agent
```

Prepare processed Search-R1 parquet files. The checked-in deceptive-search training scripts expect these files to already exist, so run this once unless you know your script handles it:

```bash
python3 examples/data_preprocess/preprocess_search_r1_dataset.py --local_dir $DATA_ROOT/searchR1_processed_direct
```

Start the retriever from the separate `retriever` environment before every deceptive-search training or evaluation run:

```bash
conda activate retriever
DATA_ROOT=$DATA_ROOT bash examples/search/retriever/retrieval_launch.sh
```

The scripts assume:

```text
http://127.0.0.1:8000/retrieve
```

For deceptive-search mitigation methods, prepare a prior deceptive-search RL actor checkpoint and set `actor_rollout_ref.model.path` in the target script, for example:

```text
checkpoints/verl_deceptive_search/grpo_deceptive_search_qwen3_8b/global_step_<N>/actor/huggingface
```

**NOTE:** Self-monitor is the exception: it initializes from a self-monitor SFT checkpoint instead of the prior environment RL checkpoint.

### RL, Verdict-Monitor, and Auxiliary Safety RLHF

#### RL

RL produces the environment RL checkpoint used by later mitigation runs:

```bash
conda activate deceptive-agent
bash examples/grpo_trainer/search_qa/run_deceptive_search.sh
```

#### Verdict-Monitor

```bash
conda activate deceptive-agent
bash examples/grpo_trainer/search_qa/run_deceptive_search_verdict_m.sh
```

Keep the retriever running. This script initializes the actor from a prior deceptive-search RL checkpoint and uses the `actor_monitor` reward manager; update `actor_rollout_ref.model.path` and `verdict_monitor_path` before launch.

#### Auxiliary Safety RLHF

Expected script path:

```text
examples/grpo_trainer/search_qa/run_deceptive_search_aux_safety.sh
```

When adding this script, keep the retriever running, initialize the actor from a prior deceptive-search RL checkpoint, and make sure the auxiliary safety data exists. If the script does not prepare auxiliary data internally, run:

```bash
python3 -m verl.trainer.auxiliary.preprocess prompt_only --dataset PKU-Alignment/PKU-SafeRLHF --split train --local_dir $DATA_ROOT/pku_safe_rlhf/prompt_only --data_source pku_safe_rlhf
```

### Self-Monitor

#### Self-Monitor SFT

If no self-monitor SFT checkpoint exists yet, train one first:

```bash
bash examples/sft/self_monitor/run_qwen2.5_7b_sp2.sh 8
```

#### Self-Monitor RL

Expected script path:

```text
examples/grpo_trainer/search_qa/run_deceptive_search_self_m.sh
```

When adding this script, keep the retriever running and initialize `actor_rollout_ref.model.path` from the self-monitor SFT checkpoint.

### Critique-Monitor and Maximin

These methods use an external critique monitor and a judge. Keep the retriever running. Use `judge_model.backend=api_cot` by default and start the CoT judge service before API-CoT runs. If using a constrained scorer, no API judge service is needed.

#### CoT Judge Service

```bash
conda activate deceptive-agent
CUDA_VISIBLE_DEVICES=6,7 bash examples/grpo_trainer/serve_cot_judge_vllm.sh 7001 2 Qwen/Qwen3-8B
```

#### Critique-Monitor

```bash
conda activate deceptive-agent
JUDGE_MODEL_NAME=Qwen/Qwen3-8B JUDGE_PORT=7001 bash examples/grpo_trainer/search_qa/run_deceptive_search_m_cot_judge.sh
```

#### Maximin

API CoT judge script:

```bash
conda activate deceptive-agent
JUDGE_MODEL_NAME=Qwen/Qwen3-8B JUDGE_PORT=7001 bash examples/grpo_trainer/search_qa/run_deceptive_search_mm_cot_judge.sh
```

Constrained-scorer variant:

```bash
bash examples/grpo_trainer/search_qa/run_deceptive_search_mm.sh
```

## CheatShop

### Shared Prep

Use the WebShop/CheatShop conda environment for data preparation, training, and API rollout evaluation:

```bash
conda activate deceptive-agent-webshop
```

Prepare CheatShop prompt rows only if the target script does not already do it:

```bash
python3 -m examples.data_preprocess.prepare --local_dir $DATA_ROOT/verl-agent --mode text --train_data_size 16 --val_data_size 32
```

The checked-in CheatShop training scripts currently call `examples.data_preprocess.prepare` internally with script-specific data sizes.

For CheatShop mitigation methods, prepare a prior CheatShop RL actor checkpoint and set `actor_rollout_ref.model.path` in the target script, for example:

```text
checkpoints/verl_agent_webshop/grpo_qwen3_8b_cheatshop/global_step_<N>/actor/huggingface
```

**NOTE:** Self-monitor is the exception: it initializes from a self-monitor SFT checkpoint instead of the prior environment RL checkpoint.

### RL, Verdict-Monitor, and Auxiliary Safety RLHF

#### RL

RL produces the environment RL checkpoint used by later mitigation runs:

```bash
conda activate deceptive-agent-webshop
bash examples/grpo_trainer/webshop/run_cheatshop.sh
```

#### Verdict-Monitor

Expected script path:

```text
examples/grpo_trainer/webshop/run_cheatshop_verdict_m.sh
```

When adding this script, run from `deceptive-agent-webshop`, initialize the actor from a prior CheatShop RL checkpoint, and use the `actor_monitor` reward manager.

#### Auxiliary Safety RLHF

Expected script path:

```text
examples/grpo_trainer/webshop/run_cheatshop_aux_safety.sh
```

When adding this script, run from `deceptive-agent-webshop`, initialize the actor from a prior CheatShop RL checkpoint, and make sure the auxiliary safety data exists. If the script does not prepare auxiliary data internally, run:

```bash
python3 -m verl.trainer.auxiliary.preprocess prompt_only --dataset PKU-Alignment/PKU-SafeRLHF --split train --local_dir $DATA_ROOT/pku_safe_rlhf/prompt_only --data_source pku_safe_rlhf
```

### Self-Monitor

#### Self-Monitor SFT

If no self-monitor SFT checkpoint exists yet, train one first:

```bash
bash examples/sft/self_monitor/run_qwen2.5_7b_sp2.sh 8
```

#### Self-Monitor RL

Expected script path:

```text
examples/grpo_trainer/webshop/run_cheatshop_self_m.sh
```

When adding this script, run from `deceptive-agent-webshop` and initialize `actor_rollout_ref.model.path` from the self-monitor SFT checkpoint.

### Critique-Monitor and Maximin

These methods use an external critique monitor and a judge. Use `judge_model.backend=api_cot` by default and start the CoT judge service before API-CoT runs. The training command itself must run from `deceptive-agent-webshop`.

#### CoT Judge Service

Start the judge from any environment with vLLM installed:

```bash
conda activate deceptive-agent
CUDA_VISIBLE_DEVICES=6,7 bash examples/grpo_trainer/serve_cot_judge_vllm.sh 7001 2 Qwen/Qwen3-8B
```

#### Critique-Monitor

```bash
conda activate deceptive-agent-webshop
JUDGE_MODEL_NAME=Qwen/Qwen3-8B JUDGE_PORT=7001 bash examples/grpo_trainer/webshop/run_cheatshop_m_cot_judge.sh
```

#### Maximin

API CoT judge script:

```bash
conda activate deceptive-agent-webshop
JUDGE_MODEL_NAME=Qwen/Qwen3-8B JUDGE_PORT=7001 bash examples/grpo_trainer/webshop/run_cheatshop_mm_cot_judge.sh
```

Constrained-scorer/Lagrangian ablation:

```bash
bash examples/grpo_trainer/webshop/run_cheatshop_mm_lag.sh
```

## API Rollout Evaluation

### Shared Evaluation Prep

Roles evaluation needs the roles parquet. Prepare it manually if the eval file is missing:

```bash
conda activate deceptive-agent
python3 examples/data_preprocess/deceptive_roles.py --local_dir $DATA_ROOT/deceptive_roles
```

Deceptive-search evaluation needs processed Search-R1 parquet files and the retriever service:

```bash
conda activate deceptive-agent
python3 examples/data_preprocess/preprocess_search_r1_dataset.py --local_dir $DATA_ROOT/searchR1_processed_direct
```

```bash
conda activate retriever
DATA_ROOT=$DATA_ROOT bash examples/search/retriever/retrieval_launch.sh
```

CheatShop evaluation must run from `deceptive-agent-webshop` and requires the WebShop/CheatShop assets installed.

### OpenAI-Compatible API

```bash
conda activate deceptive-agent
bash examples/api_rollout_eval/run_deceptive_roles_openai_api.sh
bash examples/api_rollout_eval/run_deceptive_search_openai_api.sh
```

```bash
conda activate deceptive-agent-webshop
bash examples/api_rollout_eval/run_cheatshop_openai_api.sh
```

### Local vLLM API

For roles or deceptive search, start the local server and evaluation from the main training environment:

```bash
conda activate deceptive-agent
CUDA_VISIBLE_DEVICES=0 bash examples/api_rollout_eval/serve_local_vllm.sh 7000 1 Qwen/Qwen3-8B qwen3-8b
```

```bash
bash examples/api_rollout_eval/run_deceptive_roles_local_vllm.sh model.api_base=http://127.0.0.1:7000/v1 model.model=qwen3-8b
bash examples/api_rollout_eval/run_deceptive_search_local_vllm.sh model.api_base=http://127.0.0.1:7000/v1 model.model=qwen3-8b
```

For CheatShop, run the evaluator from `deceptive-agent-webshop`:

```bash
conda activate deceptive-agent-webshop
CUDA_VISIBLE_DEVICES=0 bash examples/api_rollout_eval/serve_local_vllm.sh 7000 1 Qwen/Qwen3-8B qwen3-8b
bash examples/api_rollout_eval/run_cheatshop_local_vllm.sh model.api_base=http://127.0.0.1:7000/v1 model.model=qwen3-8b
```

## Outputs

Training checkpoints are written to:

```text
checkpoints/${trainer.project_name}/${trainer.experiment_name}
```

Hydra outputs are written to:

```text
outputs/${trainer.project_name}/${trainer.experiment_name}/${timestamp}
```

When `trainer.rollout_data_dir=auto` or `dump.output_dir=auto`, rollout dumps are written under the Hydra output directory. For trainable monitor runs, monitor checkpoints are under:

```text
checkpoints/<project>/<experiment>/global_step_<N>/monitor
```
