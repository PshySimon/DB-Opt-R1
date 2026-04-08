#!/bin/bash
# SFT 训练启动脚本（verl, 全量）
set -e

export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PROJECT_NAME="${PROJECT_NAME:-db_tuning}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-./model_save/sft_full/}"
SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-sft-full-$(basename "$BASE_MODEL")-$(date +%Y%m%d)}"
DATA_DIR="${DATA_DIR:-./datasets/sft}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
LR="${LR:-1e-6}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
N_GPUS="${N_GPUS:-2}"

if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "错误: 未找到 $DATA_DIR/train.parquet"
    echo "请先运行数据预处理:"
    echo "  python -m datasets.preprocess_sft --input_files datasets/sft/cold_start.jsonl --output_dir datasets/sft/"
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
echo "============================================================"

torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS \
    -m verl.trainer.sft_trainer \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/validation.parquet \
    data.train_batch_size=$BATCH_SIZE \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    +data.prompt_dict_keys='[question]' \
    +data.response_dict_keys='[answer]' \
    data.max_length=$MAX_LENGTH \
    optim.lr=$LR \
    model.partial_pretrain=$BASE_MODEL \
    +model.torch_dtype=bfloat16 \
    +model.attn_implementation=flash_attention_2 \
    trainer.default_local_dir=$SFT_OUTPUT_DIR \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$SFT_EXPERIMENT_NAME \
    "trainer.logger=[console]" \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=$EPOCHS
