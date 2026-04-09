#!/bin/bash
# GRPO 训练启动脚本（verl, 全量）
set -euo pipefail

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export HYDRA_FULL_ERROR=1

SFT_CHECKPOINT="${SFT_CHECKPOINT:-./model_save/sft/global_step_latest/actor}"
TRAIN_DATA="${TRAIN_DATA:-./datasets/grpo/train.parquet}"
VAL_DATA="${VAL_DATA:-./datasets/grpo/validation.parquet}"
COST_MODEL_PATH="${COST_MODEL_PATH:-./cost_model/checkpoints/v1}"
N_GPUS="${N_GPUS:-2}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
LR="${LR:-1e-6}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_REPEAT="${N_REPEAT:-5}"
TOTAL_STEPS="${TOTAL_STEPS:-100}"
SAVE_FREQ="${SAVE_FREQ:-5}"
MAX_TURNS="${MAX_TURNS:-10}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

echo "============================================"
echo "  DB-Opt GRPO 训练 (verl, 全量)"
echo "============================================"
echo "SFT Checkpoint: $SFT_CHECKPOINT"
echo "训练数据: $TRAIN_DATA"
echo "GPU 数量: $N_GPUS"
echo "N_REPEAT: $N_REPEAT"
echo "总步数: $TOTAL_STEPS"
echo "attn_impl: $ATTN_IMPL"
echo "============================================"

python3 -m training.verl.main_grpo \
  algorithm.adv_estimator=grpo \
  \
  data.train_files=$TRAIN_DATA \
  data.val_files=$VAL_DATA \
  data.train_batch_size=$BATCH_SIZE \
  data.max_prompt_length=4096 \
  data.max_response_length=4096 \
  data.max_start_length=4096 \
  data.max_tool_response_length=2048 \
  \
  actor_rollout_ref.model.path=$SFT_CHECKPOINT \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n_repeat=$N_REPEAT \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  \
  algorithm.kl_ctrl.kl_coef=0.001 \
  \
  trainer.project_name=db_tuning \
  trainer.experiment_name=grpo-full \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=1 \
  trainer.total_training_steps=$TOTAL_STEPS \
  "trainer.logger=[console]" \
  \
  tool.max_turns=$MAX_TURNS \
  +cost_model_path=$COST_MODEL_PATH
