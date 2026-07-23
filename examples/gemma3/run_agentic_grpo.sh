#!/usr/bin/env bash
set -euo pipefail

TASK=${1:?"usage: run_agentic_grpo.sh <deceptive-roles|search|deceptive-search|webshop|cheatshop> [model] [hydra overrides...]"}
shift
MODEL_PATH=${1:-google/gemma-3-4b-it}
if (($#)); then
  shift
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
case "$TASK" in
  deceptive-roles)
    TASK_SCRIPT="$REPO_ROOT/examples/grpo_trainer/deceptive_roles/run_deceptive_roles.sh"
    SCRIPT_ARGS=()
    ;;
  search)
    TASK_SCRIPT="$REPO_ROOT/examples/grpo_trainer/search_qa/run_search.sh"
    SCRIPT_ARGS=(vllm)
    ;;
  deceptive-search)
    TASK_SCRIPT="$REPO_ROOT/examples/grpo_trainer/search_qa/run_deceptive_search.sh"
    SCRIPT_ARGS=(vllm)
    ;;
  webshop)
    TASK_SCRIPT="$REPO_ROOT/examples/grpo_trainer/webshop/run_webshop.sh"
    SCRIPT_ARGS=(vllm)
    ;;
  cheatshop)
    TASK_SCRIPT="$REPO_ROOT/examples/grpo_trainer/webshop/run_cheatshop.sh"
    SCRIPT_ARGS=(vllm)
    ;;
  *)
    echo "Unsupported task: $TASK" >&2
    exit 2
    ;;
esac

# vLLM 0.8.5 V0 preserves Gemma3's bidirectional image-token attention.
export VLLM_USE_V1=0

exec bash "$TASK_SCRIPT" "${SCRIPT_ARGS[@]}" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.attn_implementation=eager \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.model.freeze_vision_tower=True \
  actor_rollout_ref.model.freeze_multi_modal_projector=True \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  "$@"
