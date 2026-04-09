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
TEST_FREQ="${TEST_FREQ:-1}"
MAX_TURNS="${MAX_TURNS:-10}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_START_LENGTH="${MAX_START_LENGTH:-4096}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-2048}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-2}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-2}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
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
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.max_start_length=$MAX_START_LENGTH \
  data.max_tool_response_length=$MAX_TOOL_RESPONSE_LENGTH \
  \
  actor_rollout_ref.model.path=$SFT_CHECKPOINT \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_SIZE \
  \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=$N_REPEAT \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.free_cache_engine=$FREE_CACHE_ENGINE \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE \
  \
  algorithm.kl_ctrl.kl_coef=0.001 \
  \
  trainer.project_name=db_tuning \
  trainer.experiment_name=grpo-full \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.total_training_steps=$TOTAL_STEPS \
  "trainer.logger=[console]" \
  \
  tool.max_turns=$MAX_TURNS \
  +cost_model_path=$COST_MODEL_PATH \
  "$@"
