#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_train_common.sh"

run_train_experiment "c2_gain_balanced" "sft_manifest_c2_gain_balanced.jsonl"
