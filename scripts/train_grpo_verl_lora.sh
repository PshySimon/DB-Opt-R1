#!/bin/bash
# GRPO 训练启动脚本（verl, LoRA）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_train_common.sh"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export HYDRA_FULL_ERROR=1

SFT_CHECKPOINT="${SFT_CHECKPOINT:-./model_save/sft/global_step_latest/actor}"
TRAIN_DATA="${TRAIN_DATA:-./datasets/grpo/train.parquet}"
VAL_DATA="${VAL_DATA:-./datasets/grpo/validation.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-./model_save/grpo_lora/}"
COST_MODEL_PATH="${COST_MODEL_PATH:-./cost_model/checkpoints/v9_lgbm}"
SCENARIO_FILES="${SCENARIO_FILES:-./data_pipeline/data/scenarios/collected/collected_server1.json,./data_pipeline/data/scenarios/collected/collected_server2.json,./data_pipeline/data/scenarios/collected/collected_server3.json}"
SCENARIO_SOURCE_FILTER="${SCENARIO_SOURCE_FILTER:-llm_generated}"
N_GPUS="${N_GPUS:-2}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
LR="${LR:-1e-6}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_REPEAT="${N_REPEAT:-5}"
TOTAL_STEPS="${TOTAL_STEPS:-100}"
SAVE_FREQ="${SAVE_FREQ:-5}"
TEST_FREQ="${TEST_FREQ:-1}"
MAX_TURNS="${MAX_TURNS:-10}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-$LORA_RANK}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
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
DEBUG_ROLLOUT_DIR="${DEBUG_ROLLOUT_DIR:-debug/rollout}"
REWARD_DEBUG_NUM_EXAMINE="${REWARD_DEBUG_NUM_EXAMINE:-0}"
VAL_REWARD_DEBUG_NUM_EXAMINE="${VAL_REWARD_DEBUG_NUM_EXAMINE:-1}"
EARLY_STOPPING_ENABLED="${EARLY_STOPPING_ENABLED:-False}"
EARLY_STOPPING_METRIC="${EARLY_STOPPING_METRIC:-val/compiler_autotuning/db_tuning/test_score}"
EARLY_STOPPING_MODE="${EARLY_STOPPING_MODE:-max}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0}"
PROJECT_NAME="${PROJECT_NAME:-db_tuning}"
GRPO_EXPERIMENT_NAME="${GRPO_EXPERIMENT_NAME:-grpo-lora}"
TRAIN_CONFIG_JSON="${TRAIN_CONFIG_JSON:-$OUTPUT_DIR/train_config.json}"

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
mkdir -p "$OUTPUT_DIR"
write_train_config_json "$TRAIN_CONFIG_JSON" \
  SFT_CHECKPOINT TRAIN_DATA VAL_DATA OUTPUT_DIR COST_MODEL_PATH SCENARIO_FILES SCENARIO_SOURCE_FILTER \
  N_GPUS CUDA_DEVICES LR BATCH_SIZE N_REPEAT TOTAL_STEPS SAVE_FREQ TEST_FREQ MAX_TURNS \
  LORA_RANK LORA_ALPHA TARGET_MODULES MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_START_LENGTH \
  MAX_TOOL_RESPONSE_LENGTH PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE \
  REF_LOG_PROB_MICRO_BATCH_SIZE ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE \
  GPU_MEMORY_UTILIZATION FREE_CACHE_ENGINE ATTN_IMPL DEBUG_ROLLOUT_DIR \
  REWARD_DEBUG_NUM_EXAMINE VAL_REWARD_DEBUG_NUM_EXAMINE \
  EARLY_STOPPING_ENABLED EARLY_STOPPING_METRIC EARLY_STOPPING_MODE \
  EARLY_STOPPING_PATIENCE EARLY_STOPPING_MIN_DELTA PROJECT_NAME \
  GRPO_EXPERIMENT_NAME TRAIN_CONFIG_JSON

echo "============================================"
echo "  DB-Opt GRPO 训练 (verl, LoRA)"
echo "============================================"
echo "SFT Checkpoint: $SFT_CHECKPOINT"
echo "训练数据: $TRAIN_DATA"
echo "输出目录: $OUTPUT_DIR"
echo "Cost Model:  $COST_MODEL_PATH"
echo "GPU 数量: $N_GPUS"
echo "N_REPEAT: $N_REPEAT"
echo "总步数: $TOTAL_STEPS"
echo "LoRA r/a: $LORA_RANK / $LORA_ALPHA"
echo "modules:  $TARGET_MODULES"
echo "attn_impl: $ATTN_IMPL"
echo "配置: $TRAIN_CONFIG_JSON"
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
  ++actor_rollout_ref.model.lora_rank=$LORA_RANK \
  ++actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
  ++actor_rollout_ref.model.target_modules=$TARGET_MODULES \
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
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.project_name=$PROJECT_NAME \
  trainer.experiment_name=$GRPO_EXPERIMENT_NAME \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.total_training_steps=$TOTAL_STEPS \
  "trainer.logger=[console]" \
  \
  tool.max_turns=$MAX_TURNS \
  debug_rollout_dir=$DEBUG_ROLLOUT_DIR \
  reward_debug_num_examine=$REWARD_DEBUG_NUM_EXAMINE \
  val_reward_debug_num_examine=$VAL_REWARD_DEBUG_NUM_EXAMINE \
  trainer.early_stopping.enabled=$EARLY_STOPPING_ENABLED \
  trainer.early_stopping.metric=$EARLY_STOPPING_METRIC \
  trainer.early_stopping.mode=$EARLY_STOPPING_MODE \
  trainer.early_stopping.patience=$EARLY_STOPPING_PATIENCE \
  trainer.early_stopping.min_delta=$EARLY_STOPPING_MIN_DELTA \
  +scenario_dir=[$SCENARIO_FILES] \
  +scenario_source_filter=$SCENARIO_SOURCE_FILTER \
  +cost_model_path=$COST_MODEL_PATH \
  "$@"
