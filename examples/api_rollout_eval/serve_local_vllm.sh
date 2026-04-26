#!/bin/bash
# Serve a local LLM with an OpenAI-compatible API using vLLM.
#
# Usage:
#   bash examples/api_rollout_eval/serve_local_vllm.sh [PORT] [TP_SIZE] [MODEL] [SERVED_MODEL_NAME]
#
# Examples:
#   bash examples/api_rollout_eval/serve_local_vllm.sh
#   bash examples/api_rollout_eval/serve_local_vllm.sh 7000 1 Qwen/Qwen2.5-7B-Instruct
#   bash examples/api_rollout_eval/serve_local_vllm.sh 7000 2 /path/to/merged_hf_checkpoint my_checkpoint
#
# Notes:
#   - Install vLLM in the environment before running this script.
#   - Set CUDA_VISIBLE_DEVICES before launching if you want specific GPUs.
#   - In the eval scripts, API_BASE must use the same PORT and MODEL_NAME must
#     match SERVED_MODEL_NAME.

set -x

PORT="${1:-7000}"
TP_SIZE="${2:-1}"
MODEL="${3:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${4:-$MODEL}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

echo "============================================================"
echo "Starting vLLM OpenAI-compatible server"
echo "============================================================"
echo "Model: $MODEL"
echo "Served model name: $SERVED_MODEL_NAME"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TP_SIZE"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "Extra vLLM Args: $EXTRA_VLLM_ARGS"
echo ""
echo "Eval API_BASE should be: http://127.0.0.1:$PORT/v1"
echo "Eval MODEL_NAME should be: $SERVED_MODEL_NAME"
echo ""

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    $EXTRA_VLLM_ARGS
