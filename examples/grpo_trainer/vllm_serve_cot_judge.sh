#!/bin/bash
# Serve a local CoT judge model with vLLM's `vllm serve` OpenAI-compatible API.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=6,7 bash examples/grpo_trainer/serve_cot_judge_vllm_serve.sh [PORT] [TP_SIZE] [MODEL] [SERVED_MODEL_NAME]
#
# Example:
#   CUDA_VISIBLE_DEVICES=6,7 bash examples/grpo_trainer/serve_cot_judge_vllm_serve.sh 7001 2 Qwen/Qwen3-8B
#
# The training script should use:
#   judge_model.api_cot.api_base=http://127.0.0.1:$PORT/v1
#   judge_model.api_cot.model=$SERVED_MODEL_NAME

set -x

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=7

PORT="${1:-7001}"
TP_SIZE="${2:-1}"
MODEL="${3:-hahnli/Qwen3-8B-CoT-Judge}"
SERVED_MODEL_NAME="${4:-Qwen3-8B-GRM}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

echo "================================================================"
echo "Starting CoT judge vLLM OpenAI-compatible server via vllm serve"
echo "================================================================"
echo "Model: $MODEL"
echo "Served model name: $SERVED_MODEL_NAME"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TP_SIZE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "Max Model Len: $MAX_MODEL_LEN"
echo "Extra vLLM Args: $EXTRA_VLLM_ARGS"
echo ""
echo "Judge API base: http://127.0.0.1:$PORT/v1"
echo "Judge API model: $SERVED_MODEL_NAME"
echo ""

vllm serve "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    $EXTRA_VLLM_ARGS
