# ============================================================
# USER CONFIGURATION — DATA_ROOT must be set before running!
# DATA_ROOT: root directory where your searchR1 data lives.
# Example:
#   DATA_ROOT=/my/data/path bash retrieval_launch.sh
# ============================================================
if [ -z "${DATA_ROOT}" ]; then
    echo "ERROR: DATA_ROOT is not set."
    echo "Please specify the root directory, which should contain the `searchR1/` data directory, e.g.:"
    echo "  DATA_ROOT=/path/to/verl_data bash examples/search/retriever/retrieval_launch.sh"
    exit 1
fi
save_path=$DATA_ROOT/searchR1

# Validate required files exist before starting the server
if [ ! -f "$save_path/e5_Flat.index" ] || [ ! -f "$save_path/wiki-18.jsonl" ]; then
    echo "ERROR: Required data files not found under: $save_path"
    echo "  Missing one or both of:"
    echo "    $save_path/e5_Flat.index"
    echo "    $save_path/wiki-18.jsonl"
    echo ""
    echo "Or run the download script first:"
    echo "  python examples/search/searchr1_download.py --local_dir \$DATA_ROOT/searchR1"
    exit 1
fi

index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python examples/search/retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --faiss_gpu \
  --port 8000 \