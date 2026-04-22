#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../_common.sh"
run_trl_sft_train_experiment "a1_full3k" "sft_manifest_a1_full3k.jsonl" "full" "$@"
