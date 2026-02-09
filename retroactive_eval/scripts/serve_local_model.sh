#!/bin/bash
# Serve a local LLM with OpenAI-compatible API using vLLM
#
# Usage:
#   ./serve_local_model.sh [model_name] [port] [tensor_parallel_size]
#
# Examples:
#   ./serve_local_model.sh  # Uses Llama-3.1-8B-Instruct on port 8000
#   ./serve_local_model.sh meta-llama/Llama-3.1-70B-Instruct 8000 4  # 4 GPUs
#   ./serve_local_model.sh /path/to/local/model 8001 1

# export CUDA_VISIBLE_DEVICES=0
PORT="${1:-8000}"
TP_SIZE="${2:-1}"
# MODEL="${3:-meta-llama/Llama-3.1-8B-Instruct}"
# MODEL="${3:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
MODEL="${3:-Qwen/Qwen2.5-72B-Instruct}"

echo "============================================================"
echo "Starting vLLM OpenAI-compatible server"
echo "============================================================"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TP_SIZE"
echo "============================================================"
echo ""
echo "Once started, use with retroactive_eval:"
echo ""
echo "  python -m retroactive_eval.run_eval \\"
echo "      --base-url http://localhost:$PORT/v1 \\"
echo "      --model $MODEL \\"
echo "      --api-key dummy"
echo ""
echo "============================================================"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization 0.98 \
    --max-model-len 4096 \
    --max-logprobs 20 \
    --dtype auto
