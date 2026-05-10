#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../_common.sh"
run_verl_rl_train_experiment "hard_1q_no_pred_v14_ge200" "$V2_RL_HARD_NO_PRED_V14_GE200" "lora" "$@"
