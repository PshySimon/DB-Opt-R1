#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../_common.sh"
run_verl_rl_train_experiment "frontier_plus_hard_1q" "$V2_RL_FRONTIER_PLUS_HARD" "full" "$@"
