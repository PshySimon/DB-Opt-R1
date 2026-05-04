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
COST_MODEL_PATH="${COST_MODEL_PATH:-./cost_model/checkpoints/v10_lgbm}"
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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
MAX_START_LENGTH="${MAX_START_LENGTH:-4096}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-2048}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-2}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-2}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
UPDATE_WEIGHTS_BUCKET_MEGABYTES="${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-3072}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
DEBUG_ROLLOUT_DIR="${DEBUG_ROLLOUT_DIR:-debug/rollout}"
GRPO_PROGRESS_LOG="${GRPO_PROGRESS_LOG:-1}"
GRPO_PROGRESS_LOG_FILE="${GRPO_PROGRESS_LOG_FILE:-$OUTPUT_DIR/progress.log}"
GRPO_PROGRESS_HEARTBEAT_INTERVAL="${GRPO_PROGRESS_HEARTBEAT_INTERVAL:-5}"
GPU_EFFICIENCY_MONITOR="${GPU_EFFICIENCY_MONITOR:-0}"
GPU_EFFICIENCY_INTERVAL="${GPU_EFFICIENCY_INTERVAL:-1}"
GPU_EFFICIENCY_LOG_FILE="${GPU_EFFICIENCY_LOG_FILE:-$OUTPUT_DIR/gpu_efficiency.log}"
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
REQUESTED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
REQUESTED_HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-}"
REQUESTED_ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-}"
GPU_EFFICIENCY_PID=""

gpu_efficiency_enabled() {
  case "$GPU_EFFICIENCY_MONITOR" in
    1|true|TRUE|True|yes|YES|Yes) return 0 ;;
    *) return 1 ;;
  esac
}

gpu_efficiency_sample_loop() {
  local query="timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"

  while true; do
    if ! nvidia-smi --query-gpu="$query" --format=csv,noheader,nounits 2>> "$GPU_EFFICIENCY_LOG_FILE" | awk -F',' '
      function trim(value) {
        gsub(/^[ \t]+|[ \t]+$/, "", value)
        return value
      }

      NF >= 7 {
        timestamp = trim($1)
        gpu_index = trim($2)
        gpu_util = trim($3) + 0
        mem_util = trim($4) + 0
        mem_used = trim($5) + 0
        mem_total = trim($6) + 0
        power_w = trim($7) + 0
        mem_used_pct = mem_total > 0 ? mem_used / mem_total * 100 : 0

        printf "%s,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n", \
          timestamp, gpu_index, gpu_util, mem_util, mem_used, mem_total, power_w, mem_used_pct
      }
    ' >> "$GPU_EFFICIENCY_LOG_FILE"; then
      echo "$(date '+%F %T') GPU_EFFICIENCY_SAMPLE_ERROR" >> "$GPU_EFFICIENCY_LOG_FILE"
    fi

    sleep "$GPU_EFFICIENCY_INTERVAL"
  done
}

append_gpu_efficiency_summary() {
  local summary_file="${GPU_EFFICIENCY_LOG_FILE}.summary.$$"

  {
    echo
    echo "=== GPU_EFFICIENCY_SUMMARY ==="
    echo "fields: gpu_util=nvidia-smi GPU-Util; mem_used_pct=memory.used/memory.total; mem_util=nvidia-smi utilization.memory"
    awk -F',' '
      function trim(value) {
        gsub(/^[ \t]+|[ \t]+$/, "", value)
        return value
      }

      /^#/ || /^===/ || /^timestamp/ || NF < 8 {
        next
      }

      {
        gpu_index = trim($2)
        gpu_util = trim($3) + 0
        mem_util = trim($4) + 0
        mem_used_gb = (trim($5) + 0) / 1024
        power_w = trim($7) + 0
        mem_used_pct = trim($8) + 0

        count[gpu_index] += 1
        gpu_util_sum[gpu_index] += gpu_util
        mem_util_sum[gpu_index] += mem_util
        mem_used_gb_sum[gpu_index] += mem_used_gb
        mem_used_pct_sum[gpu_index] += mem_used_pct
        power_w_sum[gpu_index] += power_w

        if (!(gpu_index in seen)) {
          seen[gpu_index] = 1
          gpu_util_min[gpu_index] = gpu_util
          gpu_util_max[gpu_index] = gpu_util
          mem_used_pct_min[gpu_index] = mem_used_pct
          mem_used_pct_max[gpu_index] = mem_used_pct
          mem_used_gb_min[gpu_index] = mem_used_gb
          mem_used_gb_max[gpu_index] = mem_used_gb
        }

        if (gpu_util < gpu_util_min[gpu_index]) gpu_util_min[gpu_index] = gpu_util
        if (gpu_util > gpu_util_max[gpu_index]) gpu_util_max[gpu_index] = gpu_util
        if (mem_used_pct < mem_used_pct_min[gpu_index]) mem_used_pct_min[gpu_index] = mem_used_pct
        if (mem_used_pct > mem_used_pct_max[gpu_index]) mem_used_pct_max[gpu_index] = mem_used_pct
        if (mem_used_gb < mem_used_gb_min[gpu_index]) mem_used_gb_min[gpu_index] = mem_used_gb
        if (mem_used_gb > mem_used_gb_max[gpu_index]) mem_used_gb_max[gpu_index] = mem_used_gb
      }

      END {
        gpu_count = 0
        for (gpu_index in count) {
          gpu_count += 1
          printf "GPU%s n=%d gpu_util:avg=%.1f%%,min=%.1f%%,max=%.1f%% mem_used_pct:avg=%.1f%%,min=%.1f%%,max=%.1f%% mem_used_gb:avg=%.1fGB,min=%.1fGB,max=%.1fGB mem_util:avg=%.1f%% power_w:avg=%.1fW\n", \
            gpu_index, count[gpu_index], \
            gpu_util_sum[gpu_index] / count[gpu_index], gpu_util_min[gpu_index], gpu_util_max[gpu_index], \
            mem_used_pct_sum[gpu_index] / count[gpu_index], mem_used_pct_min[gpu_index], mem_used_pct_max[gpu_index], \
            mem_used_gb_sum[gpu_index] / count[gpu_index], mem_used_gb_min[gpu_index], mem_used_gb_max[gpu_index], \
            mem_util_sum[gpu_index] / count[gpu_index], \
            power_w_sum[gpu_index] / count[gpu_index]
        }

        if (gpu_count == 0) {
          print "GPU_EFFICIENCY_SUMMARY=no_samples"
        }
      }
    ' "$GPU_EFFICIENCY_LOG_FILE"
  } > "$summary_file"

  cat "$summary_file" >> "$GPU_EFFICIENCY_LOG_FILE"
  rm -f "$summary_file"
}

start_gpu_efficiency_monitor() {
  if ! gpu_efficiency_enabled; then
    echo "[gpu_efficiency] disabled"
    return 0
  fi

  mkdir -p "$(dirname "$GPU_EFFICIENCY_LOG_FILE")"
  {
    echo "# GPU_EFFICIENCY_LOG_FILE=$GPU_EFFICIENCY_LOG_FILE"
    echo "# gpu_util = nvidia-smi GPU-Util"
    echo "# mem_used_pct = memory.used / memory.total"
    echo "# mem_util = nvidia-smi utilization.memory, i.e. memory bandwidth utilization"
    echo "=== GPU_EFFICIENCY_RAW_SAMPLES ==="
    echo "timestamp,index,gpu_util,mem_util,mem_used_mib,mem_total_mib,power_w,mem_used_pct"
  } > "$GPU_EFFICIENCY_LOG_FILE"

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU_EFFICIENCY_SUMMARY=no_nvidia_smi" >> "$GPU_EFFICIENCY_LOG_FILE"
    echo "[gpu_efficiency] nvidia-smi not found, log_file=$GPU_EFFICIENCY_LOG_FILE"
    return 0
  fi

  gpu_efficiency_sample_loop &
  GPU_EFFICIENCY_PID="$!"
  echo "[gpu_efficiency] sampling pid=$GPU_EFFICIENCY_PID log_file=$GPU_EFFICIENCY_LOG_FILE interval_s=$GPU_EFFICIENCY_INTERVAL"
}

finish_gpu_efficiency_monitor() {
  local exit_code=$?
  trap - EXIT
  set +e

  if gpu_efficiency_enabled; then
    if [ -n "$GPU_EFFICIENCY_PID" ]; then
      kill "$GPU_EFFICIENCY_PID" >/dev/null 2>&1
      wait "$GPU_EFFICIENCY_PID" >/dev/null 2>&1
    fi

    if [ -f "$GPU_EFFICIENCY_LOG_FILE" ]; then
      append_gpu_efficiency_summary
      echo "[gpu_efficiency] summary_file=$GPU_EFFICIENCY_LOG_FILE"
    else
      echo "[gpu_efficiency] no_log_file=$GPU_EFFICIENCY_LOG_FILE"
    fi
  fi

  exit "$exit_code"
}

N_GPUS="$(infer_n_gpus "$CUDA_DEVICES" "$N_GPUS")"
configure_accelerator_visible_devices \
  "$CUDA_DEVICES" \
  "$N_GPUS" \
  "$REQUESTED_CUDA_VISIBLE_DEVICES" \
  "$REQUESTED_HIP_VISIBLE_DEVICES" \
  "$REQUESTED_ROCR_VISIBLE_DEVICES"
mkdir -p "$OUTPUT_DIR"
write_train_config_json "$TRAIN_CONFIG_JSON" \
  SFT_CHECKPOINT TRAIN_DATA VAL_DATA OUTPUT_DIR COST_MODEL_PATH SCENARIO_FILES SCENARIO_SOURCE_FILTER \
  N_GPUS CUDA_DEVICES CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES \
  LR BATCH_SIZE N_REPEAT TOTAL_STEPS SAVE_FREQ TEST_FREQ MAX_TURNS \
  LORA_RANK LORA_ALPHA TARGET_MODULES MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_MODEL_LEN \
  MAX_START_LENGTH MAX_TOOL_RESPONSE_LENGTH PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE \
  REF_LOG_PROB_MICRO_BATCH_SIZE ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE \
  GPU_MEMORY_UTILIZATION UPDATE_WEIGHTS_BUCKET_MEGABYTES FREE_CACHE_ENGINE ATTN_IMPL DEBUG_ROLLOUT_DIR \
  GRPO_PROGRESS_LOG GRPO_PROGRESS_LOG_FILE GRPO_PROGRESS_HEARTBEAT_INTERVAL \
  GPU_EFFICIENCY_MONITOR GPU_EFFICIENCY_INTERVAL GPU_EFFICIENCY_LOG_FILE \
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
echo "max_model_len: $MAX_MODEL_LEN"
echo "update bucket MB: $UPDATE_WEIGHTS_BUCKET_MEGABYTES"
echo "attn_impl: $ATTN_IMPL"
echo "progress log: $GRPO_PROGRESS_LOG_FILE"
echo "gpu efficiency monitor: $GPU_EFFICIENCY_MONITOR"
echo "gpu efficiency log: $GPU_EFFICIENCY_LOG_FILE"
echo "配置: $TRAIN_CONFIG_JSON"
echo "============================================"

export GRPO_PROGRESS_LOG GRPO_PROGRESS_LOG_FILE GRPO_PROGRESS_HEARTBEAT_INTERVAL

trap finish_gpu_efficiency_monitor EXIT
start_gpu_efficiency_monitor

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
  data.use_custom_tool_format_func=True \
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
  actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=$UPDATE_WEIGHTS_BUCKET_MEGABYTES \
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
