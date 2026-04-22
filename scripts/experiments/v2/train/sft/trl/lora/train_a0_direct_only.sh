#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../_common.sh"
run_trl_sft_train_experiment "a0_direct_only" "sft_manifest_a0_direct_only.jsonl" "lora" "$@"
