#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_train_common.sh"

run_train_experiment "b2_depth_full" "sft_manifest_b2_depth_full.jsonl"
