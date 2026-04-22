#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../_common.sh"
run_trl_sft_train_experiment "c1_gain_natural" "sft_manifest_c1_gain_natural.jsonl" "full" "$@"
