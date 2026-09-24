# Experiment Kickoff Guide

This is the shortest path from a fresh checkout to the repository's main research experiments. The codebase combines multi-turn LLM-RL with reward models, unauthorized-tool environments, several monitor designs, and game-theoretic agent–monitor training.

## Pick an environment

| Paper name | What it studies | Runtime name | Guide |
| --- | --- | --- | --- |
| Agentic Role-Playing | Reward hacking and deception under conflicting roles | `ReasonChat` | [agentic_role_playing.md](agentic_role_playing.md) |
| SearchQA | QA with public search, optionally with an unauthorized answer tool | `search` / `deceptive_search` | [searchqa.md](searchqa.md) |
| WebShop | Shopping, optionally with unauthorized analysis/search tools | `Webshop` / `CheatShop` | [webshop.md](webshop.md) |

Each environment page gives setup, one data command, the main launch matrix, and evaluation commands. [methods.md](methods.md) explains RL, Mixed Safety RLHF, Self-Monitor, Critique Monitor, Verdict Monitor, and e-PoG.

## Before any launch

Complete the root [Installation guide](../README.md#installation), then run commands from the repository root. The shell files are experiment specifications, not portable wrappers: inspect their first lines and replace hard-coded `DATA_ROOT`, checkpoint paths, GPU visibility, offline Hugging Face settings, and W&B settings. Most main launches use eight GPUs.

```bash
conda activate deceptive-agent
pip install -e .
```

WebShop needs its separate environment; SearchQA needs a retriever process. Follow the corresponding environment page first.

## Recommended path

1. Launch the plain RL baseline for one environment.
2. Read [methods.md](methods.md) and select a monitoring baseline or e-PoG.
3. Use the environment page's matching script. Suffixes such as `_qwen4b` and `_gemma3` select model-specific reproductions; the unsuffixed script is the primary Qwen3-8B entry point.
4. Use `examples/api_rollout_eval/` for checkpoint or API evaluation.

The root [README.md](../README.md) describes the broader framework.
