#!/usr/bin/env bash
set -euo pipefail

# Prepare the RTX PRO 6000 Blackwell / sm_120 LLaMA-Factory training environment.
#
# Defaults match the LoRA + FlashAttention-2 smoke that was validated on 2026-04-30:
#   torch            2.8.0+cu128
#   transformers     4.52.4
#   datasets         3.6.0
#   accelerate       1.7.0
#   deepspeed        0.16.9
#   llamafactory     0.9.3
#   flash-attn       2.8.3 prebuilt wheel
#
# Usage:
#   bash scripts/setup_sm120_llamafactory_env.sh
#   SKIP_GPU_CHECK=true bash scripts/setup_sm120_llamafactory_env.sh
#   RUN_SMOKE=true bash scripts/setup_sm120_llamafactory_env.sh
#   SMOKE_ONLY=true RUN_SMOKE=true bash scripts/setup_sm120_llamafactory_env.sh

ENV_PREFIX="${ENV_PREFIX:-/root/autodl-tmp/conda_envs/dbopt-lf-sm120}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/DB-Opt-R1}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen3-8B}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data_pipeline/data/train/v3/full_v3_b_step_no_think_history_3k}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
RECREATE="${RECREATE:-false}"
RUN_SMOKE="${RUN_SMOKE:-false}"
SMOKE_ONLY="${SMOKE_ONLY:-false}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-false}"

PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.52.4}"
DATASETS_VERSION="${DATASETS_VERSION:-3.6.0}"
ACCELERATE_VERSION="${ACCELERATE_VERSION:-1.7.0}"
DEEPSPEED_VERSION="${DEEPSPEED_VERSION:-0.16.9}"
LLAMAFACTORY_VERSION="${LLAMAFACTORY_VERSION:-0.9.3}"
PEFT_VERSION="${PEFT_VERSION:-0.15.2}"
TRL_VERSION="${TRL_VERSION:-0.9.6}"
TOKENIZERS_VERSION="${TOKENIZERS_VERSION:-0.21.1}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"

SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-$PROJECT_ROOT/model_save/experiments/sm120/smoke_lf_lora_1step_fa2}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
SMOKE_LOG="${SMOKE_LOG:-$LOG_DIR/sm120_lf_lora_1step_fa2_$(date +%Y%m%d_%H%M%S).log}"
SMOKE_MAX_LENGTH="${SMOKE_MAX_LENGTH:-512}"
SMOKE_CUDA_DEVICES="${SMOKE_CUDA_DEVICES:-0}"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

log() {
  printf '[sm120-lf-env] %s\n' "$*"
}

die() {
  printf '[sm120-lf-env][ERROR] %s\n' "$*" >&2
  exit 1
}

require_bool() {
  case "$2" in
    true|false) ;;
    *) die "$1 must be true or false, got: $2" ;;
  esac
}

require_bool RECREATE "$RECREATE"
require_bool RUN_SMOKE "$RUN_SMOKE"
require_bool SMOKE_ONLY "$SMOKE_ONLY"
require_bool SKIP_GPU_CHECK "$SKIP_GPU_CHECK"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [ ! -f "$CONDA_SH" ] && [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
fi
[ -f "$CONDA_SH" ] || die "Cannot find conda activation script. Set CONDA_SH=/path/to/conda.sh"

log "Conda env: $ENV_PREFIX"
log "Project:   $PROJECT_ROOT"
log "Model:     $BASE_MODEL"
log "Data:      $DATA_DIR"
log "Torch arch: $TORCH_CUDA_ARCH_LIST"

# shellcheck source=/dev/null
source "$CONDA_SH"

if [ -d "$ENV_PREFIX" ] && [ "$RECREATE" = "true" ]; then
  [ "$SMOKE_ONLY" = "false" ] || die "RECREATE=true cannot be used with SMOKE_ONLY=true"
  log "Removing existing env because RECREATE=true"
  conda env remove -p "$ENV_PREFIX" -y
fi

if [ ! -d "$ENV_PREFIX" ]; then
  [ "$SMOKE_ONLY" = "false" ] || die "SMOKE_ONLY=true requires an existing env: $ENV_PREFIX"
  log "Creating conda env"
  conda create -p "$ENV_PREFIX" "python=$PYTHON_VERSION" -y
else
  log "Using existing conda env"
fi

conda activate "$ENV_PREFIX"

if [ "$SMOKE_ONLY" = "false" ]; then
  log "Upgrading installer basics"
  python -m pip install -U pip setuptools wheel packaging ninja

  log "Installing PyTorch CUDA 12.8 stack"
  python -m pip install --no-cache-dir \
    "torch==$TORCH_VERSION" torchvision torchaudio \
    --index-url "$PYTORCH_INDEX_URL"

  log "Installing LLaMA-Factory training stack"
  python -m pip install --no-cache-dir \
    "transformers==$TRANSFORMERS_VERSION" \
    "datasets==$DATASETS_VERSION" \
    "accelerate==$ACCELERATE_VERSION" \
    "deepspeed==$DEEPSPEED_VERSION" \
    "peft==$PEFT_VERSION" \
    "trl==$TRL_VERSION" \
    "tokenizers==$TOKENIZERS_VERSION" \
    "llamafactory[deepspeed,metrics]==$LLAMAFACTORY_VERSION" \
    -i "$PYPI_INDEX_URL"

  log "Installing FlashAttention-2 from prebuilt wheel"
  python -m pip uninstall -y flash-attn || true
  python -m pip install --no-cache-dir "$FLASH_ATTN_WHEEL_URL"
else
  log "SMOKE_ONLY=true, skipping package installation"
fi

log "Verifying training package versions"
SKIP_GPU_CHECK="$SKIP_GPU_CHECK" \
TORCH_VERSION="$TORCH_VERSION" \
TRANSFORMERS_VERSION="$TRANSFORMERS_VERSION" \
DATASETS_VERSION="$DATASETS_VERSION" \
ACCELERATE_VERSION="$ACCELERATE_VERSION" \
DEEPSPEED_VERSION="$DEEPSPEED_VERSION" \
LLAMAFACTORY_VERSION="$LLAMAFACTORY_VERSION" \
FLASH_ATTN_VERSION="$FLASH_ATTN_VERSION" \
python - <<'PY'
import os
import torch
import transformers
import datasets
import accelerate
import deepspeed
import llamafactory
import flash_attn

print("torch =", torch.__version__, torch.version.cuda)
print("transformers =", transformers.__version__)
print("datasets =", datasets.__version__)
print("accelerate =", accelerate.__version__)
print("deepspeed =", deepspeed.__version__)
print("llamafactory =", llamafactory.__version__)
print("flash_attn =", flash_attn.__version__)

if not torch.__version__.startswith(os.environ["TORCH_VERSION"]):
    raise SystemExit(f"expected torch {os.environ['TORCH_VERSION']}, got {torch.__version__}")
if torch.version.cuda != "12.8":
    raise SystemExit(f"expected CUDA 12.8 torch runtime, got {torch.version.cuda}")
if transformers.__version__ != os.environ["TRANSFORMERS_VERSION"]:
    raise SystemExit(f"unexpected transformers version: {transformers.__version__}")
if datasets.__version__ != os.environ["DATASETS_VERSION"]:
    raise SystemExit(f"unexpected datasets version: {datasets.__version__}")
if accelerate.__version__ != os.environ["ACCELERATE_VERSION"]:
    raise SystemExit(f"unexpected accelerate version: {accelerate.__version__}")
if deepspeed.__version__ != os.environ["DEEPSPEED_VERSION"]:
    raise SystemExit(f"unexpected deepspeed version: {deepspeed.__version__}")
if llamafactory.__version__ != os.environ["LLAMAFACTORY_VERSION"]:
    raise SystemExit(f"unexpected llamafactory version: {llamafactory.__version__}")
if flash_attn.__version__ != os.environ["FLASH_ATTN_VERSION"]:
    raise SystemExit(f"unexpected flash_attn version: {flash_attn.__version__}")

if os.environ.get("SKIP_GPU_CHECK") != "true":
    assert torch.cuda.is_available(), "CUDA is not available"
    capability = torch.cuda.get_device_capability(0)
    print("gpu =", torch.cuda.get_device_name(0), capability)
    assert capability[0] == 12, f"expected sm_120 class GPU, got {capability}"
PY

log "Checking llamafactory-cli"
command -v llamafactory-cli >/dev/null || die "llamafactory-cli is not in PATH"
llamafactory-cli version || true

if [ "$RUN_SMOKE" = "true" ]; then
  [ -d "$PROJECT_ROOT" ] || die "PROJECT_ROOT does not exist: $PROJECT_ROOT"
  [ -d "$BASE_MODEL" ] || die "BASE_MODEL does not exist: $BASE_MODEL"
  [ -d "$DATA_DIR" ] || die "DATA_DIR does not exist: $DATA_DIR"
  compgen -G "$DATA_DIR/*.jsonl" >/dev/null || die "No train JSONL shards found under $DATA_DIR"

  mkdir -p "$LOG_DIR"
  if [ -d "$SMOKE_OUTPUT_DIR" ]; then
    log "Removing old smoke output: $SMOKE_OUTPUT_DIR"
    rm -rf "$SMOKE_OUTPUT_DIR"
  fi
  log "Generating LLaMA-Factory LoRA smoke config"

  PROJECT_ROOT="$PROJECT_ROOT" \
  BASE_MODEL="$BASE_MODEL" \
  DATA_DIR="$DATA_DIR" \
  OUTPUT_DIR="$SMOKE_OUTPUT_DIR" \
  CUDA_DEVICES="$SMOKE_CUDA_DEVICES" \
  N_GPUS=1 \
  MAX_LENGTH="$SMOKE_MAX_LENGTH" \
  MAX_STEPS=1 \
  SAVE_STRATEGY=steps \
  SAVE_STEPS=999999 \
  SAVE_TOTAL_LIMIT=1 \
  PER_DEVICE_BATCH_SIZE=1 \
  GRAD_ACCUM=1 \
  PLOT_LOSS=false \
  FLASH_ATTN=fa2 \
  DEEPSPEED_CONFIG="" \
  DRY_RUN=true \
  "$PROJECT_ROOT/scripts/train_sft_llamafactory_full.sh"

  python - <<PY
from pathlib import Path

p = Path("$SMOKE_OUTPUT_DIR/llamafactory_train.yaml")
s = p.read_text(encoding="utf-8")
s = s.replace("finetuning_type: full", "finetuning_type: lora")
s = s.replace("save_strategy: steps", 'save_strategy: "no"')
s = "\n".join(line for line in s.splitlines() if not line.startswith("deepspeed:")) + "\n"
s += "lora_rank: 8\n"
s += "lora_alpha: 16\n"
s += "lora_dropout: 0.0\n"
s += "lora_target: all\n"
p.write_text(s, encoding="utf-8")
PY

  log "Running LLaMA-Factory LoRA + FlashAttention-2 smoke"
  CUDA_VISIBLE_DEVICES="$SMOKE_CUDA_DEVICES" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  llamafactory-cli train "$SMOKE_OUTPUT_DIR/llamafactory_train.yaml" 2>&1 | tee "$SMOKE_LOG"

  grep -q "Using FlashAttention-2 for faster training and inference" "$SMOKE_LOG" \
    || die "LLaMA-Factory did not select FlashAttention-2. See $SMOKE_LOG"
  grep -q "Fine-tuning method: LoRA" "$SMOKE_LOG" \
    || die "LoRA smoke was not observed. See $SMOKE_LOG"
  grep -q "train_loss" "$SMOKE_LOG" \
    || die "Smoke train_loss not found. See $SMOKE_LOG"

  log "Smoke passed: $SMOKE_LOG"
fi
