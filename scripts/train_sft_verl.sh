#!/bin/bash
# SFT 训练启动脚本
#
# 使用 verl 的 fsdp_sft_trainer 进行 SFT 训练。
# 参考 Agent-R1 的 train_Exp_1_pureSFT.sh。
#
# 使用前：
#   1. 准备 JSONL 轨迹文件（cold_start.jsonl / trajectories.jsonl）
#   2. 运行预处理：
#      python -m datasets.preprocess_sft \
#          --input_files datasets/sft/cold_start.jsonl \
#          --output_dir datasets/sft/
#   3. 运行本脚本

set -e

export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# ============================================================
# 配置
# ============================================================
base_model="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
project_name="db_tuning"
sft_output_dir="./model_save/sft/"
sft_experiment_name="sft-$(basename $base_model)-$(date +%Y%m%d)"

data_dir="./datasets/sft"
max_length="${MAX_LENGTH:-8192}"
lr="${LR:-1e-6}"
epochs="${EPOCHS:-3}"
batch_size="${BATCH_SIZE:-16}"
micro_batch_size="${MICRO_BATCH_SIZE:-4}"
n_gpus="${N_GPUS:-2}"

# ============================================================
# 检查数据
# ============================================================
if [ ! -f "$data_dir/train.parquet" ]; then
    echo "错误: 未找到 $data_dir/train.parquet"
    echo "请先运行数据预处理:"
    echo "  python -m datasets.preprocess_sft --input_files datasets/sft/cold_start.jsonl --output_dir datasets/sft/"
    exit 1
fi

echo "============================================================"
echo "SFT 训练配置"
echo "============================================================"
echo "基座模型:     $base_model"
echo "数据目录:     $data_dir"
echo "输出目录:     $sft_output_dir"
echo "max_length:   $max_length"
echo "lr:           $lr"
echo "epochs:       $epochs"
echo "batch_size:   $batch_size"
echo "GPU 数量:     $n_gpus"
echo "============================================================"

# ============================================================
# 训练
# ============================================================
torchrun --standalone --nnodes=1 --nproc_per_node=$n_gpus \
    -m verl.trainer.sft_trainer \
    data.train_files=$data_dir/train.parquet \
    data.val_files=$data_dir/validation.parquet \
    data.train_batch_size=$batch_size \
    data.micro_batch_size_per_gpu=$micro_batch_size \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    +data.prompt_dict_keys='[question]' \
    +data.response_dict_keys='[answer]' \
    data.max_length=$max_length \
    optim.lr=$lr \
    model.partial_pretrain=$base_model \
    +model.torch_dtype=bfloat16 \
    +model.attn_implementation=flash_attention_2 \
    trainer.default_local_dir=$sft_output_dir \
    trainer.project_name=$project_name \
    trainer.experiment_name=$sft_experiment_name \
    "trainer.logger=[console]" \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=$epochs

echo "============================================================"
echo "训练完成! Checkpoint 保存在: $sft_output_dir"
echo "============================================================"
