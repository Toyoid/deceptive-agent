#!/bin/bash
# Serve a local CoT judge model with an OpenAI-compatible API using vLLM.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=6,7 bash examples/grpo_trainer/serve_cot_judge_vllm.sh [PORT] [TP_SIZE] [MODEL] [SERVED_MODEL_NAME]
#
# Example:
#   CUDA_VISIBLE_DEVICES=6,7 bash examples/grpo_trainer/serve_cot_judge_vllm.sh 7001 2 Qwen/Qwen3-14B
#
# The training script should use:
#   judge_model.api_cot.api_base=http://127.0.0.1:$PORT/v1
#   judge_model.api_cot.model=$SERVED_MODEL_NAME

set -x

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=6,7

PORT="${1:-7001}"
TP_SIZE="${2:-2}"
MODEL="${3:-Qwen/Qwen3-8B}"
SERVED_MODEL_NAME="${4:-$MODEL}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.87}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

echo "============================================================"
echo "Starting CoT judge vLLM OpenAI-compatible server"
echo "============================================================"
echo "Model: $MODEL"
echo "Served model name: $SERVED_MODEL_NAME"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TP_SIZE"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "Max Model Len: $MAX_MODEL_LEN"
echo "Extra vLLM Args: $EXTRA_VLLM_ARGS"
echo ""
echo "Judge API base: http://127.0.0.1:$PORT/v1"
echo "Judge API model: $SERVED_MODEL_NAME"
echo ""

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    $EXTRA_VLLM_ARGS
