set -x

RM_MODEL=${1:-sfairXC/FsfairX-LLaMA3-RM-v0.1}

DATASET=PKU-Alignment/PKU-SafeRLHF
SPLIT=test
DATA_ROOT=/devsft_AFS/hanxiaoli/verl_data

PAIR_DIR=$DATA_ROOT/pku_safe_rlhf/rm_paired
PAIR_DATA=$PAIR_DIR/$SPLIT.parquet
RESULT_DIR=../outputs/pku_rm_eval

# Step 1: convert PKU-style response pairs into the auxiliary RM-eval parquet.
python3 -m verl.trainer.auxiliary.preprocess \
    pku_rm_eval \
    --dataset $DATASET \
    --split $SPLIT \
    --local_dir $PAIR_DIR \
    --data_source pku_safe_rlhf

# Step 2: evaluate whether the adopted RM prefers the annotated safer/better response.
python3 -m verl.trainer.auxiliary.eval_rm_accuracy \
    --data_path $PAIR_DATA \
    --model $RM_MODEL \
    --batch_size 8 \
    --dtype bfloat16 \
    --device auto \
    --tie_policy half \
    --output_json $RESULT_DIR/${SPLIT}_metrics.json
