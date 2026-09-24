# Deceptive Agent

Deceptive Agent is specifically designed to study **emergent deception** and its **mitigation** in LLM agents under **RL training** and **diverse monitoring mechanisms**. It provides complete training and evaluation pipelines, controllable agentic environments, and most importantly, **a practical implementation of max-min game-theoretic approach for agent-monitor co-learning**.

## Key Features

| Area | Available Support |
| --- | --- |
| **Controllable Deception-Study Environments** | ✅ Agentic Role-Playing<br>✅ SearchQA w/ or wo/ unauthorized information uses <br>✅ WebShop w/ or wo/ unauthorized tools<br>✅ AppWorld (experimental) |
| **Alignment Methods** | ✅ RL<br>✅ Mixed Safety RLHF<br>✅ Self-Monitor<br>✅ Verdict Monitor<br>✅ Critique Monitor<br>✅ e-PoG |
| **e-PoG Training** | ✅ Max–min agent–monitor co-optimization<br>✅ Weak-to-strong oversight recipes |
| **Verifier Backends** | ✅ OpenAI-compatible CoT verifier<br>✅ logit post-processing verifier<br>✅ Configurable score profiles and verifier prompts |
| **Reward Integration** | ✅ Actor–monitor reward manager<br>✅ Lagrangian dual updates & Constrained RL<br> |
| **RL Algorithms** | ✅ GRPO<br>✅ PPO<br>✅ DAPO<br>✅ RLOO<br>✅ REINFORCE++ and more |
| **Models** | ✅ Qwen3-4B, Qwen3-8B, Gemma-3-4B-IT recipes and more<br>✅ LoRA training support |
| **Rollout & Scaling** | ✅ vLLM and SGLang rollouts<br>✅ Multi-turn tool interaction<br>✅ Parallel and grouped environments<br>✅ Ray + FSDP distributed training<br>✅ Independent actor, monitor, and verifier GPU placement |
| **Evaluation & Analysis** | ✅ Local-vLLM and OpenAI-compatible API rollouts<br>✅ Deception- and environment-specific metrics |

## e-PoG: A Game-Theoretic Agent–Monitor Co-Learning Framework for Deception Mitigation

Static monitors can become unreliable as an RL agent learns which behaviors evade their oversight. **Evaluation-Side Partially Observable Extensive-Form Game (e-PoG)** instead formulates deception mitigation as a game between an agent and a trainable monitor. The agent maximizes task utility while paying a trust penalty for verified deceptive behavior; the monitor reviews the agent's trajectory, produces a structured critique, and learns from a separate verifier's score. Co-learning continually exposes both policies to the other's evolving strategy rather than treating oversight as a fixed reward signal. See our paper for the full algorithmic design and rigorous theoretical guarantees of e-PoG.

<p align="center">
  <img src="docs/figures/e-pog-overview.png" width="100%" alt="Overview of the e-PoG agent-monitor co-learning framework">
</p>

The following shows the training dynamics and representative cases of emergent deceptive behaviors in various alignment approaches, and deception mitigation under e-PoG.

<p align="center">
  <img src="docs/figures/e-pog-deception-dynamics.png" width="100%" alt="Deception dynamics during e-PoG training"><br>
  <img src="docs/figures/e-pog-case-study.png" width="100%" alt="Representative e-PoG case study">
</p>

## Installation

### Install the core training stack

The project is built on [veRL](https://github.com/volcengine/verl) and [verl-agent](https://github.com/langfengQ/verl-agent). A Linux machine with NVIDIA GPUs, CUDA 12.4-compatible drivers, and Conda is recommended. The commands below match the versions pinned by this repository.

```bash
conda create -n deceptive-agent python=3.12 -y
conda activate deceptive-agent
cd /path/to/deceptive-agent

pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.5.post1
```

> [!IMPORTANT]
> Install WebShop and the SearchQA retriever in dedicated Conda environments. Their dependency constraints conflict with the main training stack.

### Install supported environments

#### 1. Agentic Role-Playing

Agentic Role-Playing (`ReasonChat`) has no additional service dependency. Its data is included in the repository and is preprocessed in the main environment:

```bash
conda activate deceptive-agent
cd /path/to/deceptive-agent
python3 examples/data_preprocess/deceptive_roles.py \
  --local_dir "$DATA_ROOT/deceptive_roles"
```

See the [Agentic Role-Playing kickoff guide](experiment_kickoff_guide/agentic_role_playing.md) for launch scripts and expected checkpoints.

#### 2. WebShop

WebShop requires Python 3.10. Create a separate environment, install the simulator assets, and then install this repository into that environment:

```bash
conda create -n deceptive-agent-webshop python=3.10 -y
conda activate deceptive-agent-webshop
cd /path/to/deceptive-agent

cd agent_system/environments/env_package/webshop/webshop
bash setup.sh -d all
cd /path/to/deceptive-agent

pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.5.post1
```

If `gdown` cannot fetch the WebShop assets, provide a Google Drive cookie in `.cache/gdown/cookies.txt` or download the assets manually. Dependency-resolver warnings about WebShop's older `spacy`/`typer` constraints can be ignored when the environment imports and setup complete successfully.

This installation supports both standard WebShop and the unauthorized-tool setting (`CheatShop`).

#### 3. SearchQA

Install the inherited Search environment package into the main training environment:

```bash
conda activate deceptive-agent
cd /path/to/deceptive-agent
pip install -e agent_system/environments/env_package/search/third_party
pip install gym==0.26.2
```

Prepare the SearchQA parquet files:

```bash
cd /path/to/deceptive-agent
python3 examples/data_preprocess/preprocess_search_r1_dataset.py \
  --local_dir "$DATA_ROOT/searchR1_processed_direct"
```

The retrieval server uses FAISS GPU and should run in a separate Python 3.10 environment. It uses roughly 6 GB of GPU memory; reserve capacity for it when assigning training GPUs.

```bash
conda create -n retriever python=3.10 -y
conda activate retriever

conda install numpy==1.26.4 -y
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets pyserini huggingface_hub uvicorn fastapi
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y
```

Download and assemble the E5 index and Wikipedia corpus:

```bash
conda activate retriever
cd /path/to/deceptive-agent
python examples/search/searchr1_download.py --local_dir "$DATA_ROOT/searchR1"
cat "$DATA_ROOT"/searchR1/part_* > "$DATA_ROOT/searchR1/e5_Flat.index"
gzip -d "$DATA_ROOT/searchR1/wiki-18.jsonl.gz"
```

Start the retriever before SearchQA training or evaluation. Redirecting its output avoids terminal I/O latency spikes:

```bash
cd /path/to/deceptive-agent
DATA_ROOT="$DATA_ROOT" \
  bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log 2>&1
```

The default endpoint is `http://127.0.0.1:8000/retrieve`.

#### 4. AppWorld (experimental)

AppWorld support is experimental. Install the client integration in the main environment:

```bash
conda activate deceptive-agent
cd /path/to/deceptive-agent
pip install git+https://github.com/StonyBrookNLP/appworld.git
appworld install
pip install -e .
pip install vllm==0.8.5.post1
```

Run AppWorld services from a dedicated environment:

```bash
conda create -n appworld python=3.12 -y
conda activate appworld
pip install git+https://github.com/StonyBrookNLP/appworld.git
appworld install
appworld download data
```

The experimental multi-instance launcher is `examples/env_server/start_appworld_server.sh`. Review its batch sizes, port range, and Conda initialization before use.

## Quick Start

The reference e-PoG configurations jointly run an actor, a trainable monitor, environment workers, and a CoT verifier. Our full-scale runs use the following single-node resources:

| Model | Recommended hardware |
| --- | --- |
| Qwen3-8B | 8 × NVIDIA A800 80GB |
| Qwen3-4B or Gemma-3-4B-IT | 8 × NVIDIA L40S 48GB |

The primary scripts reserve GPUs `0-6` for training and GPU `7` for the verifier. Before launching, replace their machine-specific `DATA_ROOT`, actor, monitor, reward-model, and checkpoint paths. SearchQA and WebShop e-PoG runs should initialize the actor from the corresponding environment RL checkpoint. See the [Experiment Kickoff Guide](experiment_kickoff_guide/README.md) for data preparation and model-specific variants.

### Prepare the CoT verifier

Start the verifier in a separate terminal after installing the main environment. The served name must match `JUDGE_MODEL_NAME` in the training commands.

```bash
conda activate deceptive-agent
cd /path/to/deceptive-agent

CUDA_VISIBLE_DEVICES=7 \
  bash examples/grpo_trainer/vllm_serve_cot_judge.sh \
  7001 1 hahnli/Qwen3-8B-CoT-Judge Qwen3-8B-GRM
```

Wait until the OpenAI-compatible endpoint is ready at `http://127.0.0.1:7001/v1`.

### Agentic Role-Playing: e-PoG

The script prepares the role-playing data and uses a reward model for task utility with a Lagrangian trust constraint.

```bash
conda activate deceptive-agent
cd /path/to/deceptive-agent

JUDGE_MODEL_NAME=Qwen3-8B-GRM JUDGE_PORT=7001 \
  bash examples/grpo_trainer/deceptive_roles/run_deceptive_roles_mm_lag_cot_judge.sh
```

### SearchQA with unauthorized tools: e-PoG

Start the retriever first and leave it running:

```bash
conda activate retriever
cd /path/to/deceptive-agent

DATA_ROOT=/path/to/verl_data \
  bash examples/search/retriever/retrieval_launch.sh
```

Then launch e-PoG from another terminal:

```bash
conda activate deceptive-agent
cd /path/to/deceptive-agent

JUDGE_MODEL_NAME=Qwen3-8B-GRM JUDGE_PORT=7001 \
  bash examples/grpo_trainer/search_qa/run_deceptive_search_mm_cot_judge.sh
```

### WebShop with unauthorized tools: e-PoG

Run training in the dedicated WebShop environment while keeping the verifier in its original terminal:

```bash
conda activate deceptive-agent-webshop
cd /path/to/deceptive-agent

JUDGE_MODEL_NAME=Qwen3-8B-GRM JUDGE_PORT=7001 \
  bash examples/grpo_trainer/webshop/run_cheatshop_mm_cot_judge.sh
```

For the complete experiment index, method mapping, and model-specific scripts, see the [Experiment Kickoff Guide](experiment_kickoff_guide/README.md).

## Acknowledgments

Deceptive Agent benefits from the foundations established by [veRL](https://github.com/volcengine/verl) and [verl-agent](https://github.com/langfengQ/verl-agent), and the environments are adapted from [SkyRL-Gym](https://github.com/NovaSky-AI/SkyRL), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), and [WebShop](https://github.com/princeton-nlp/WebShop); we thank the authors and contributors of these projects for their important contributions to the open-source research community.
