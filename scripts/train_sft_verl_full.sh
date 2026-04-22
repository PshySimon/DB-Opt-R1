#!/bin/bash
# SFT 训练启动脚本（verl, 全量）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_train_common.sh"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PROJECT_NAME="${PROJECT_NAME:-db_tuning}"
OUTPUT_DIR="${OUTPUT_DIR:-${SFT_OUTPUT_DIR:-./model_save/sft_full/}}"
SFT_OUTPUT_DIR="$OUTPUT_DIR"
SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-sft-full-$(basename "$BASE_MODEL")-$(date +%Y%m%d)}"
DATA_DIR="${DATA_DIR:-./datasets/sft_cleaned}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
LR="${LR:-1e-6}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
N_GPUS="${N_GPUS:-2}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
TRAIN_CONFIG_JSON="${TRAIN_CONFIG_JSON:-$SFT_OUTPUT_DIR/train_config.json}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
TORCHRUN_PORT="${TORCHRUN_PORT:-${MASTER_PORT:-}}"
TORCHRUN_RUN_ID="${TORCHRUN_RUN_ID:-}"

mkdir -p "$SFT_OUTPUT_DIR"
TORCHRUN_PORT="$(infer_torchrun_port "$TORCHRUN_PORT")"
TORCHRUN_RUN_ID="$(infer_torchrun_run_id "$TORCHRUN_RUN_ID")"
export MASTER_ADDR TORCHRUN_PORT TORCHRUN_RUN_ID
export MASTER_PORT="$TORCHRUN_PORT"
write_train_config_json "$TRAIN_CONFIG_JSON" \
    BASE_MODEL PROJECT_NAME SFT_OUTPUT_DIR SFT_EXPERIMENT_NAME DATA_DIR MAX_LENGTH \
    LR EPOCHS BATCH_SIZE MICRO_BATCH_SIZE N_GPUS CUDA_VISIBLE_DEVICES \
    HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES \
    ATTN_IMPL MASTER_ADDR MASTER_PORT TORCHRUN_PORT TORCHRUN_RUN_ID TRAIN_CONFIG_JSON

if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "错误: 未找到 $DATA_DIR/train.parquet"
    echo "请先运行数据预处理:"
    echo "  python -m data_pipeline.preprocess_sft --input_files data_pipeline/data/train/sft_trajectories_cleaned.jsonl --output_dir ./datasets/sft_cleaned/"
    exit 1
fi

echo "============================================================"
echo "SFT 训练配置 (verl, 全量)"
echo "============================================================"
echo "基座模型:     $BASE_MODEL"
echo "数据目录:     $DATA_DIR"
echo "输出目录:     $SFT_OUTPUT_DIR"
echo "max_length:   $MAX_LENGTH"
echo "lr:           $LR"
echo "epochs:       $EPOCHS"
echo "batch_size:   $BATCH_SIZE"
echo "GPU 数量:     $N_GPUS"
echo "attn_impl:    $ATTN_IMPL"
echo "Rdzv:         $MASTER_ADDR:$TORCHRUN_PORT ($TORCHRUN_RUN_ID)"
echo "配置:         $TRAIN_CONFIG_JSON"
echo "============================================================"

torchrun \
    --nnodes=1 \
    --nproc_per_node=$N_GPUS \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$TORCHRUN_PORT" \
    --rdzv-id="$TORCHRUN_RUN_ID" \
    -m verl.trainer.sft_trainer \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/validation.parquet \
    data.train_batch_size=$BATCH_SIZE \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    data.max_length=$MAX_LENGTH \
    data.ignore_input_ids_mismatch=True \
    optim.lr=$LR \
    model.path=$BASE_MODEL \
    trainer.default_local_dir=$SFT_OUTPUT_DIR \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$SFT_EXPERIMENT_NAME \
    "trainer.logger=[console]" \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=$EPOCHS
