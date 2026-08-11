#!/bin/bash
# Serve a local LLM with OpenAI-compatible API using vLLM

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PORT="${1:-6000}"
TP_SIZE="${2:-8}"
# MODEL="${3:-/ssd/work/models/Qwen3-235B-A22B-Instruct-2507}"
MODEL="${3:-Qwen/Qwen2.5-72B-Instruct}"
SERVED_MODEL_NAME="${4:-Qwen-Instruct-Large}"

echo "============================================================"
echo "Starting vLLM OpenAI-compatible server"
echo "============================================================"
echo "Model: $MODEL"
echo "Served model name: $SERVED_MODEL_NAME"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TP_SIZE"
echo "============================================================"
echo ""
echo "Once started, use with retroactive_eval:"
echo ""
echo "  python -m retroactive_eval.run_eval \\"
echo "      --base-url http://localhost:$PORT/v1 \\"
echo "      --model $SERVED_MODEL_NAME \\"
echo "      --api-key dummy"
echo ""
echo "============================================================"
echo ""

vllm serve "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization 0.80 \
    --max-model-len 5120 \
    --dtype auto \
    --max-logprobs 20 \
