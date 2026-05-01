#!/usr/bin/env bash
set -euo pipefail

# Prepare the RTX PRO 6000 Blackwell / sm_120 vLLM + FlashInfer environment.
#
# Defaults match the environment validated on 2026-05-01:
#   torch                  2.11.0+cu130
#   vllm                   0.20.0
#   transformers           4.57.6
#   flashinfer-python      0.6.9
#   flashinfer-cubin       0.6.9
#   flashinfer-jit-cache   0.6.9+cu130
#
# Usage:
#   bash scripts/setup_sm120_vllm_flashinfer_env.sh
#   RUN_SMOKE=true bash scripts/setup_sm120_vllm_flashinfer_env.sh
#   RECREATE=true RUN_SMOKE=true bash scripts/setup_sm120_vllm_flashinfer_env.sh

ENV_PREFIX="${ENV_PREFIX:-/root/autodl-tmp/conda_envs/dbopt-vllm-flashinfer-cu130}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/DB-Opt-R1}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen3-8B}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
RECREATE="${RECREATE:-false}"
RUN_SMOKE="${RUN_SMOKE:-false}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-false}"

VLLM_INDEX_URL="${VLLM_INDEX_URL:-https://wheels.vllm.ai/cu130}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
VLLM_VERSION="${VLLM_VERSION:-0.20.0}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.6.9}"
FLASHINFER_CUDA="${FLASHINFER_CUDA:-cu130}"
FLASHINFER_INDEX_URL="${FLASHINFER_INDEX_URL:-https://flashinfer.ai/whl/${FLASHINFER_CUDA}}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.57.6}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
SMOKE_SCRIPT="${SMOKE_SCRIPT:-$PROJECT_ROOT/tmp/sm120_vllm_offline_flashinfer_smoke.py}"
SMOKE_LOG="${SMOKE_LOG:-$LOG_DIR/sm120_vllm_flashinfer_attention_config_$(date +%Y%m%d_%H%M%S).log}"

export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.0f}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

log() {
  printf '[sm120-env] %s\n' "$*"
}

die() {
  printf '[sm120-env][ERROR] %s\n' "$*" >&2
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
require_bool SKIP_GPU_CHECK "$SKIP_GPU_CHECK"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [ ! -f "$CONDA_SH" ] && [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
fi
[ -f "$CONDA_SH" ] || die "Cannot find conda activation script. Set CONDA_SH=/path/to/conda.sh"

log "Conda env: $ENV_PREFIX"
log "Project:   $PROJECT_ROOT"
log "Model:     $BASE_MODEL"
log "FlashInfer arch: $FLASHINFER_CUDA_ARCH_LIST"

# shellcheck source=/dev/null
source "$CONDA_SH"

if [ -d "$ENV_PREFIX" ] && [ "$RECREATE" = "true" ]; then
  log "Removing existing env because RECREATE=true"
  conda env remove -p "$ENV_PREFIX" -y
fi

if [ ! -d "$ENV_PREFIX" ]; then
  log "Creating conda env"
  conda create -p "$ENV_PREFIX" "python=$PYTHON_VERSION" -y
else
  log "Using existing conda env"
fi

conda activate "$ENV_PREFIX"

log "Upgrading installer basics"
python -m pip install -U pip setuptools wheel

log "Installing vLLM from CUDA 13.0 wheel index"
python -m pip install --no-cache-dir \
  "vllm==$VLLM_VERSION" \
  --index-url "$VLLM_INDEX_URL" \
  --extra-index-url "$PYPI_INDEX_URL"

log "Pinning transformers"
python -m pip install --no-cache-dir \
  "transformers==$TRANSFORMERS_VERSION" \
  -i "$PYPI_INDEX_URL"

log "Installing FlashInfer ${FLASHINFER_VERSION} ${FLASHINFER_CUDA} packages"
python -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache || true

python -m pip install --no-cache-dir --no-deps --force-reinstall \
  "flashinfer-python==$FLASHINFER_VERSION" \
  "flashinfer-cubin==$FLASHINFER_VERSION" \
  -i "$PYPI_INDEX_URL"

python -m pip install --no-cache-dir --no-deps --force-reinstall \
  "flashinfer-jit-cache==${FLASHINFER_VERSION}+${FLASHINFER_CUDA}" \
  --index-url "$FLASHINFER_INDEX_URL"

log "Verifying package versions"
SKIP_GPU_CHECK="$SKIP_GPU_CHECK" FLASHINFER_CUDA="$FLASHINFER_CUDA" VLLM_VERSION="$VLLM_VERSION" python - <<'PY'
import os
import torch
from importlib.metadata import version

pkgs = {
    "vllm": version("vllm"),
    "transformers": version("transformers"),
    "flashinfer-python": version("flashinfer-python"),
    "flashinfer-cubin": version("flashinfer-cubin"),
    "flashinfer-jit-cache": version("flashinfer-jit-cache"),
}

print("torch =", torch.__version__, torch.version.cuda)
print(pkgs)

if not str(torch.version.cuda).startswith("13."):
    raise SystemExit(f"expected CUDA 13.x torch runtime, got {torch.version.cuda}")
if pkgs["vllm"] != os.environ["VLLM_VERSION"]:
    raise SystemExit(f"expected vllm {os.environ['VLLM_VERSION']}, got {pkgs['vllm']}")

base = pkgs["flashinfer-jit-cache"].split("+", 1)[0]
cuda_suffix = os.environ["FLASHINFER_CUDA"]
assert pkgs["flashinfer-python"] == base, pkgs
assert pkgs["flashinfer-cubin"] == base, pkgs
assert pkgs["flashinfer-jit-cache"].endswith("+" + cuda_suffix), pkgs

if os.environ.get("SKIP_GPU_CHECK") != "true":
    assert torch.cuda.is_available(), "CUDA is not available"
    capability = torch.cuda.get_device_capability(0)
    print("gpu =", torch.cuda.get_device_name(0), capability)
    assert capability[0] == 12, f"expected sm_120 class GPU, got {capability}"

import flashinfer
print("flashinfer import ok:", getattr(flashinfer, "__version__", "unknown"))
PY

log "Verifying FlashInfer module registration"
flashinfer show-config

if [ "$RUN_SMOKE" = "true" ]; then
  [ -d "$PROJECT_ROOT" ] || die "PROJECT_ROOT does not exist: $PROJECT_ROOT"
  [ -d "$BASE_MODEL" ] || die "BASE_MODEL does not exist: $BASE_MODEL"

  mkdir -p "$LOG_DIR" "$(dirname "$SMOKE_SCRIPT")"
  log "Writing vLLM FlashInfer smoke script: $SMOKE_SCRIPT"

  cat > "$SMOKE_SCRIPT" <<PY
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def main():
    model = "$BASE_MODEL"
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "/no_think\\n一句话说明 shared_buffers 是什么。"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    llm = LLM(
        model=model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=$MAX_MODEL_LEN,
        gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,
        trust_remote_code=True,
        attention_config=AttentionConfig(
            backend=AttentionBackendEnum.FLASHINFER,
        ),
    )

    outs = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=128))
    print("=== OUTPUT ===")
    print(outs[0].outputs[0].text)


if __name__ == "__main__":
    main()
PY

  log "Running vLLM FlashInfer smoke"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  FLASHINFER_CUDA_ARCH_LIST="$FLASHINFER_CUDA_ARCH_LIST" \
  TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_LOGGING_LEVEL=INFO \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  python "$SMOKE_SCRIPT" 2>&1 | tee "$SMOKE_LOG"

  grep -q "Using AttentionBackendEnum.FLASHINFER backend" "$SMOKE_LOG" \
    || die "vLLM did not select FlashInfer attention backend. See $SMOKE_LOG"
  grep -q "Warming up FlashInfer attention" "$SMOKE_LOG" \
    || die "FlashInfer attention warmup was not observed. See $SMOKE_LOG"
  grep -q "=== OUTPUT ===" "$SMOKE_LOG" \
    || die "Smoke output marker not found. See $SMOKE_LOG"

  log "Smoke passed: $SMOKE_LOG"
else
  log "RUN_SMOKE=false, skipped model-loading vLLM smoke"
fi

log "Environment is ready"
