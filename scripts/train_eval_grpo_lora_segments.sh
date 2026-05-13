#!/bin/bash
# Train verl GRPO LoRA in checkpoint segments, eval each checkpoint, then resume.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

SFT_CHECKPOINT="${SFT_CHECKPOINT:-/root/workspace/DB-Opt-R1/model_save/experiments/v3/sft/llamafactory/full/full_v3_b_step_no_think_history_3k_match_a800_batch_noeval/checkpoint-633}"
TRAIN_JSONL="${TRAIN_JSONL:-/root/workspace/DB-Opt-R1/data_pipeline/data/train/v2/rl/rl_frontier_1q_no_pred_v10_ge200.jsonl}"
DATA_DIR="${DATA_DIR:-/root/workspace/DB-Opt-R1/datasets/rl_v2/frontier_1q_no_pred_v10_ge200}"
TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/train.parquet}"
VAL_DATA="${VAL_DATA:-$DATA_DIR/validation.parquet}"
SCENARIO_FILES="${SCENARIO_FILES:-/root/workspace/DB-Opt-R1/data_pipeline/data/scenarios/collected/collected_8c16g_hdd_20k.json}"
SCENARIO_SOURCE_FILTER="${SCENARIO_SOURCE_FILTER:-llm_generated}"
COST_MODEL_PATH="${COST_MODEL_PATH:-/root/workspace/DB-Opt-R1/cost_model/checkpoints/v10_lgbm}"
KNOB_SPACE="${KNOB_SPACE:-/root/workspace/DB-Opt-R1/configs/knob_space.yaml}"
EVAL_QUESTIONS="${EVAL_QUESTIONS:-/root/workspace/DB-Opt-R1/data_pipeline/data/eval/v2/eval_trajectories_v2.jsonl}"
EVAL_SCENARIOS="${EVAL_SCENARIOS:-/root/workspace/DB-Opt-R1/data_pipeline/data/eval/v2/collected_eval_v2.json}"

RUN_ID="${RUN_ID:-frontier_1q_no_pred_v10_no_sync_gmem028_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/workspace/DB-Opt-R1/model_save/experiments/v2/rl/verl/lora/${RUN_ID}}"
EVAL_ROOT="${EVAL_ROOT:-/root/private_data/workspace/DB-Opt-R1/eval_results/v2/rl/${RUN_ID}}"
MERGED_ROOT="${MERGED_ROOT:-/root/workspace/DB-Opt-R1/model_save/eval_merged/${RUN_ID}}"

START_STEP="${START_STEP:-10}"
END_STEP="${END_STEP:-50}"
STEP_INTERVAL="${STEP_INTERVAL:-10}"
SAVE_FREQ="${SAVE_FREQ:-$STEP_INTERVAL}"
TEST_FREQ="${TEST_FREQ:-0}"
MAX_TURNS="${MAX_TURNS:-10}"
EVAL_PARALLEL="${EVAL_PARALLEL:-64}"
API_MAX_CONCURRENT="${API_MAX_CONCURRENT:-64}"
PORT="${PORT:-8010}"
GPU_UTIL_SERVE="${GPU_UTIL_SERVE:-0.85}"
MAX_MODEL_LEN_SERVE="${MAX_MODEL_LEN_SERVE:-16384}"
KEEP_MERGED="${KEEP_MERGED:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
EVAL_PYTHON="${EVAL_PYTHON:-/root/private_data/workspace/conda_envs/dbopt-eval/bin/python}"
RL_VAL_RATIO="${RL_VAL_RATIO:-0.1}"
RL_DATA_SEED="${RL_DATA_SEED:-42}"
FORCE_REBUILD_DATA="${FORCE_REBUILD_DATA:-false}"

N_GPUS="${N_GPUS:-4}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
TP_SERVE="${TP_SERVE:-$N_GPUS}"
LR="${LR:-1e-6}"
BATCH_SIZE="${BATCH_SIZE:-4}"
N_REPEAT="${N_REPEAT:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.28}"
UPDATE_WEIGHTS_BUCKET_MEGABYTES="${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-3072}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-1}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
MAX_START_LENGTH="${MAX_START_LENGTH:-4096}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-2048}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-$LORA_RANK}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-4}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}"
ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}"
ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-False}"
ACTOR_USE_TORCH_COMPILE="${ACTOR_USE_TORCH_COMPILE:-False}"
REF_USE_TORCH_COMPILE="${REF_USE_TORCH_COMPILE:-False}"

VLLM_PID=""

stop_vllm() {
  if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    kill "$VLLM_PID" >/dev/null 2>&1 || true
    wait "$VLLM_PID" >/dev/null 2>&1 || true
  fi
  VLLM_PID=""
}

cleanup() {
  stop_vllm
}
trap cleanup EXIT

prepare_training_data() {
  if [ "$FORCE_REBUILD_DATA" = "true" ] || [ ! -f "$TRAIN_DATA" ] || [ ! -f "$VAL_DATA" ]; then
    echo "[segment] build GRPO parquet: $DATA_DIR"
    test -f "$TRAIN_JSONL"
    mkdir -p "$DATA_DIR"
    IFS=',' read -r -a scenario_args <<< "$SCENARIO_FILES"
    PYTHONPATH="$PROJECT_ROOT" \
    python -m data_pipeline.preprocess_grpo \
      --input-files "$TRAIN_JSONL" \
      --scenarios "${scenario_args[@]}" \
      --source-filter "$SCENARIO_SOURCE_FILTER" \
      --output-dir "$DATA_DIR" \
      --val-ratio "$RL_VAL_RATIO" \
      --seed "$RL_DATA_SEED" \
      --questions-per-scene 1
  fi
}

check_inputs() {
  test -x "$EVAL_PYTHON"
  test -d "$SFT_CHECKPOINT"
  test -f "$TRAIN_JSONL"
  test -d "$COST_MODEL_PATH"
  test -f "$KNOB_SPACE"
  test -f "$EVAL_QUESTIONS"
  test -f "$EVAL_SCENARIOS"
}

wait_for_vllm() {
  local port="$1"
  "$EVAL_PYTHON" - "$port" <<'PY'
import sys
import time
import urllib.request

port = sys.argv[1]
url = f"http://127.0.0.1:{port}/v1/models"
last_error = None
for i in range(180):
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            print(f"vLLM ready: {resp.status}")
            raise SystemExit(0)
    except Exception as exc:
        last_error = exc
        if i % 10 == 0:
            print(f"waiting {i}: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(1)
print(f"vLLM did not become ready: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
}

train_to_step() {
  local step="$1"
  local ckpt_dir="$OUTPUT_DIR/global_step_${step}"

  if [ -d "$ckpt_dir/actor/lora_adapter" ]; then
    echo "[segment] checkpoint exists, skip training: $ckpt_dir"
    return 0
  fi

  echo "[segment] train until global_step_${step}"
  DBOPT_BLOCK_SYNC_COMMIT=1 \
  SFT_CHECKPOINT="$SFT_CHECKPOINT" \
  TRAIN_DATA="$TRAIN_DATA" \
  VAL_DATA="$VAL_DATA" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  COST_MODEL_PATH="$COST_MODEL_PATH" \
  SCENARIO_FILES="$SCENARIO_FILES" \
  SCENARIO_SOURCE_FILTER="$SCENARIO_SOURCE_FILTER" \
  N_GPUS="$N_GPUS" \
  CUDA_DEVICES="$CUDA_DEVICES" \
  LR="$LR" \
  BATCH_SIZE="$BATCH_SIZE" \
  N_REPEAT="$N_REPEAT" \
  TOTAL_STEPS="$step" \
  SAVE_FREQ="$SAVE_FREQ" \
  TEST_FREQ="$TEST_FREQ" \
  MAX_TURNS="$MAX_TURNS" \
  LORA_RANK="$LORA_RANK" \
  LORA_ALPHA="$LORA_ALPHA" \
  TARGET_MODULES="$TARGET_MODULES" \
  MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
  MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  MAX_START_LENGTH="$MAX_START_LENGTH" \
  MAX_TOOL_RESPONSE_LENGTH="$MAX_TOOL_RESPONSE_LENGTH" \
  PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
  PPO_MICRO_BATCH_SIZE="$PPO_MICRO_BATCH_SIZE" \
  REF_LOG_PROB_MICRO_BATCH_SIZE="$REF_LOG_PROB_MICRO_BATCH_SIZE" \
  ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE="$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE" \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  UPDATE_WEIGHTS_BUCKET_MEGABYTES="$UPDATE_WEIGHTS_BUCKET_MEGABYTES" \
  FREE_CACHE_ENGINE="$FREE_CACHE_ENGINE" \
  PROJECT_NAME="${PROJECT_NAME:-db_tuning}" \
  GRPO_EXPERIMENT_NAME="${GRPO_EXPERIMENT_NAME:-$RUN_ID}" \
  GRPO_PROGRESS_LOG_FILE="$OUTPUT_DIR/progress.log" \
  bash "$PROJECT_ROOT/scripts/train_grpo_verl_lora.sh" \
    trainer.val_before_train=False \
    trainer.resume_mode=auto \
    actor_rollout_ref.rollout.agent.num_workers="$ROLLOUT_AGENT_NUM_WORKERS" \
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.enable_chunked_prefill="$ROLLOUT_ENABLE_CHUNKED_PREFILL" \
    actor_rollout_ref.rollout.enable_prefix_caching="$ROLLOUT_ENABLE_PREFIX_CACHING" \
    actor_rollout_ref.actor.use_torch_compile="$ACTOR_USE_TORCH_COMPILE" \
    actor_rollout_ref.ref.use_torch_compile="$REF_USE_TORCH_COMPILE"

  test -d "$ckpt_dir/actor/lora_adapter"
}

merge_checkpoint() {
  local step="$1"
  local adapter="$OUTPUT_DIR/global_step_${step}/actor/lora_adapter"
  local merged="$MERGED_ROOT/global_step_${step}_merged_hf"

  mkdir -p "$MERGED_ROOT"
  "$EVAL_PYTHON" "$PROJECT_ROOT/scripts/merge_lora_checkpoint.py" \
    --base-model "$SFT_CHECKPOINT" \
    --adapter "$adapter" \
    --output-dir "$merged" \
    --device-map auto \
    --overwrite
  echo "$merged"
}

eval_checkpoint() {
  local step="$1"
  local merged="$2"
  local model_name="${RUN_ID}_global_step_${step}"
  local out="$EVAL_ROOT/global_step_${step}/$(date +%Y%m%d_%H%M%S)"
  local report="$out/report/eval_report.json"

  if [ "$FORCE_EVAL" != "1" ] && find "$EVAL_ROOT/global_step_${step}" -path '*/report/eval_report.json' -type f 2>/dev/null | grep -q .; then
    echo "[segment] eval already exists, skip: $EVAL_ROOT/global_step_${step}"
    return 0
  fi

  mkdir -p "$out/rollouts" "$out/report"

  echo "[segment] serve vLLM for global_step_${step}"
  stop_vllm
  NO_PROXY=127.0.0.1,localhost,::1,0.0.0.0 \
  PATH="$(dirname "$EVAL_PYTHON"):$PATH" \
  MODEL_PATH="$merged" \
  PORT="$PORT" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN_SERVE" \
  GPU_UTIL="$GPU_UTIL_SERVE" \
  TP="$TP_SERVE" \
  SERVED_MODEL_NAME="$model_name" \
  bash "$PROJECT_ROOT/scripts/serve_vllm.sh" &
  VLLM_PID="$!"
  wait_for_vllm "$PORT"

  echo "[segment] eval global_step_${step}"
  NO_PROXY=127.0.0.1,localhost,::1,0.0.0.0 \
  DBOPT_BLOCK_SYNC_COMMIT=1 \
  PYTHONPATH="$PROJECT_ROOT" \
  "$EVAL_PYTHON" -m data_pipeline.synthesis.trajectory.sampler \
    --mode eval \
    --eval-questions "$EVAL_QUESTIONS" \
    --scenarios "$EVAL_SCENARIOS" \
    --cost-model "$COST_MODEL_PATH" \
    --knob-space "$KNOB_SPACE" \
    --output-dir "$out/rollouts" \
    --output-file sft_trajectories.jsonl \
    --model "$model_name" \
    --api-base "http://127.0.0.1:${PORT}/v1" \
    --api-key EMPTY \
    --parallel "$EVAL_PARALLEL" \
    --api-max-concurrent "$API_MAX_CONCURRENT" \
    --max-turns "$MAX_TURNS"

  PYTHONPATH="$PROJECT_ROOT" \
  "$EVAL_PYTHON" -m evaluate.run \
    --eval-data "$out/rollouts/sft_trajectories.jsonl" \
    --scenarios "$EVAL_SCENARIOS" \
    --cost-model "$COST_MODEL_PATH" \
    --knob-space "$KNOB_SPACE" \
    --output "$out/report" \
    --skip-bo

  stop_vllm
  echo "[segment] report: $report"
}

echo "============================================"
echo " segmented GRPO LoRA train + eval"
echo " RUN_ID=$RUN_ID"
echo " OUTPUT_DIR=$OUTPUT_DIR"
echo " EVAL_ROOT=$EVAL_ROOT"
echo " TRAIN_JSONL=$TRAIN_JSONL"
echo " TRAIN_DATA=$TRAIN_DATA"
echo " COST_MODEL_PATH=$COST_MODEL_PATH"
echo " TRAIN_GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo " TRAIN_BATCH_SIZE=$BATCH_SIZE"
echo " TRAIN_N_REPEAT=$N_REPEAT"
echo " TRAIN_MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo " TRAIN_MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH"
echo " SERVE_GPU_UTIL=$GPU_UTIL_SERVE"
echo " TP_SERVE=$TP_SERVE"
echo " DBOPT_BLOCK_SYNC_COMMIT=1"
echo " steps: $START_STEP..$END_STEP / $STEP_INTERVAL"
echo "============================================"

check_inputs
prepare_training_data

step="$START_STEP"
while [ "$step" -le "$END_STEP" ]; do
  train_to_step "$step"
  merged_model="$(merge_checkpoint "$step" | tail -n 1)"
  eval_checkpoint "$step" "$merged_model"
  if [ "$KEEP_MERGED" != "1" ]; then
    rm -rf "$merged_model"
    echo "[segment] removed merged model: $merged_model"
  fi
  step=$((step + STEP_INTERVAL))
done

echo "[segment] all done"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "EVAL_ROOT=$EVAL_ROOT"
