#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../_common.sh"
run_verl_sft_train_experiment "c2_gain_balanced" "sft_manifest_c2_gain_balanced.jsonl" "lora" "$@"
