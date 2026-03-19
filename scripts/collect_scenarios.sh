#!/bin/bash
# SFT 场景数据采集
# 将 knob 配置应用到 PG → pgbench → 采集完整指标
# 默认后台执行

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/common.sh"

# 默认参数
INPUT="datasets/data/scenarios/knob_configs_8c16g_hdd.json"
OUTPUT="datasets/data/scenarios/collected.json"
KNOB_SPACE="configs/knob_space.yaml"

usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "SFT 场景数据采集：应用 knob → pgbench → 采集指标"
    echo "默认后台执行，日志输出到 logs/scenarios/"
    echo ""
    echo "选项:"
    print_pg_help
    echo "  --input        knob 配置文件 (默认: $INPUT)"
    echo "  --output       输出文件 (默认: $OUTPUT)"
    echo "  --config       knob_space 配置 (默认: $KNOB_SPACE)"
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
        --input)        INPUT="$2"; shift 2 ;;
        --output)       OUTPUT="$2"; shift 2 ;;
        --config)       KNOB_SPACE="$2"; shift 2 ;;
        --foreground)   FOREGROUND=true; shift ;;
        -h|--help)      usage ;;
        *)              echo "未知参数: $1"; usage ;;
    esac
done

cd "$PROJECT_DIR"

# 统计配置数量
if [ -f "$INPUT" ]; then
    COUNT=$(python3 -c "import json; print(len(json.load(open('$INPUT'))))" 2>/dev/null || echo "?")
else
    echo "错误: 找不到输入文件 $INPUT"
    exit 1
fi

echo "=========================================="
echo "  SFT 场景数据采集"
echo "=========================================="
print_pg_info
echo "输入:      ${INPUT} (${COUNT} 条)"
echo "输出:      ${OUTPUT}"
echo ""

check_dependencies
check_pg_ready
check_database

# 构建命令
CMD="python3 -m datasets.synthesis.scenarios.pipeline collect \
    --input $INPUT \
    --output $OUTPUT \
    --config $KNOB_SPACE \
    --host $PG_HOST --port $PG_PORT \
    --user $PG_USER \
    --database $PG_DATABASE"

[ -n "$PG_PASSWORD" ] && CMD="$CMD --password $PG_PASSWORD"
[ -n "$PG_DATA_DIR" ] && CMD="$CMD --pg-data-dir $PG_DATA_DIR"

if [ "$FOREGROUND" = true ]; then
    eval $CMD
else
    run_in_background "logs/scenarios" "$CMD"
fi
