# Repo-Specific Reminders

- Before proposing, designing, or implementing changes in this repository, first fully understand both:
  - the existing codebase design and data flow
  - the user's design intention for the feature or behavior
- Do not infer interaction semantics, notation choices, or prompt layering casually. Verify how the current codebase models them first, then align the implementation with the user's intended design.

# Agent Notes For This Repo

- This repository is developed locally and run on a remote server.
- Do not install heavy packages in the local development environment, such as `torch`.
- Lightweight packages are allowed when needed for local development and validation, such as `pytest` and other small Python-derived utilities.

## Architecture Overview of This Repository
The system is a three-player max-min alignment game:

1. Assistant (π_L) — generates responses to user queries; trained via PPO/GRPO to maximize task reward while minimizing trust-penalty cost (Lagrangian formulation)
2. Monitor (π_M) — trained adversarially to expose honesty issues in the assistant's responses; outputs structured <critique>...</critique> tags; its reward is the trust-penalty score (judge validity score)
3. Judge (J) — a frozen model that scores each monitor critique for validity via constrained single-token scoring (0/1/2/3 → weighted to [0,1])

Data Flow:
```
Assistant generates response
    → Environment returns (reward, evidence, monitor_text)
    → Monitor generates <critique> tags from the history
    → Judge scores each critique independently (constrained token, max_new_tokens=1)
    → Per-critique scores aggregated via mean → trust_penalty ∈ [0,1]
    → trust_penalty becomes the "cost" in a Lagrangian PPO objective
    → λ (dual variable) adapts via SGD on the constraint violation
```

The key training signals:
- Actor: advantage = (reward_advantage - λ * cost_advantage) / (1 + λ) (ray_trainer.py:1910)
- Monitor: GRPO on trust_penalty as reward (via MonitorRewardManager)
- Judge: Frozen, not trained

Key entry files:
- verl\trainer\config\ppo_trainer.yaml
- verl\trainer\main_ppo.py
- verl\trainer\ppo\ray_trainer.py
- agent_system\multi_turn_rollout\rollout_loop.py
- agent_system\environments\env_manager.py