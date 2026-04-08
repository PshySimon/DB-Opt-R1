#!/bin/bash
# 兼容入口：默认走 LoRA 版本
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/train_grpo_trl_lora.sh"
