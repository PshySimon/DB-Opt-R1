#!/bin/bash
# Cost Model 数据采集
# 随机采样 knob → pgbench → 采集指标，输出 CSV
# 默认后台执行

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/common.sh"

# 默认参数
OUTPUT_DIR="${OUTPUT_DIR:-datasets/data/cost_model}"
ROUNDS=100
SAMPLING="random"
WORKLOAD="all"

usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "Cost Model 数据采集：随机采样 knob → pgbench → 采集指标"
    echo "默认后台执行，日志输出到 logs/costmodel/"
    echo ""
    echo "选项:"
    print_pg_help
    echo "  --output       输出目录 (默认: $OUTPUT_DIR)"
    echo "  --rounds       采集轮数 (默认: $ROUNDS)"
    echo "  --sampling     采样策略: random/lhs (默认: $SAMPLING)"
    echo "  --workload     负载类型: mixed/read_only/high_concurrency/write_heavy/all (默认: all)"
    echo "  --foreground   前台执行（调试用）"
    echo "  -h, --help     显示帮助"
    exit 0
}

# 解析 PG 参数
eval "$(parse_pg_args "$@")"

# 解析剩余参数
FOREGROUND=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --output)       OUTPUT_DIR="$2"; shift 2 ;;
        --rounds)       ROUNDS="$2"; shift 2 ;;
        --sampling)     SAMPLING="$2"; shift 2 ;;
        --workload)     WORKLOAD="$2"; shift 2 ;;
        --foreground)   FOREGROUND=true; shift ;;
        -h|--help)      usage ;;
        *)              echo "未知参数: $1"; usage ;;
    esac
done

cd "$PROJECT_DIR"

echo "=========================================="
echo "  Cost Model 数据采集"
echo "=========================================="
print_pg_info
echo "输出目录:  ${OUTPUT_DIR}"
echo "采集轮数:  ${ROUNDS}"
echo "采样策略:  ${SAMPLING}"
echo ""

check_dependencies
check_pg_ready
check_database

# 构建命令
CMD="python3 -m cost_model.data.pipeline \
    --config configs/knob_space.yaml \
    --host $PG_HOST --port $PG_PORT \
    --user $PG_USER \
    --database $PG_DATABASE \
    --output $OUTPUT_DIR \
    --rounds $ROUNDS \
    --sampling $SAMPLING \
    --workload $WORKLOAD"

[ -n "$PG_PASSWORD" ] && CMD="$CMD --password $PG_PASSWORD"
[ -n "$PG_DATA_DIR" ] && CMD="$CMD --pg-data-dir $PG_DATA_DIR"

if [ "$FOREGROUND" = true ]; then
    eval $CMD
else
    run_in_background "logs/costmodel" "$CMD"
fi
