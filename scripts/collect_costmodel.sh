#!/bin/bash
# Cost Model 数据采集
# 随机采样 knob → pgbench → 采集指标，输出 CSV
# 默认后台执行

set -e

# 默认参数
PG_HOST="${PG_HOST:-127.0.0.1}"
PG_PORT="${PG_PORT:-5432}"
PG_USER="${PG_USER:-postgres}"
PG_PASSWORD="${PG_PASSWORD:-}"
PG_DATABASE="${PG_DATABASE:-benchmark}"
PG_DATA_DIR="${PG_DATA_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-datasets/data/cost_model}"
ROUNDS="${ROUNDS:-100}"
SAMPLING="${SAMPLING:-random}"
WORKLOAD="${WORKLOAD:-all}"
FOREGROUND=false

usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "Cost Model 数据采集：随机采样 knob → pgbench → 采集指标"
    echo "默认后台执行，日志和 PID 保存到 logs/costmodel/"
    echo ""
    echo "选项:"
    echo "  --host         PG 主机地址 (默认: $PG_HOST)"
    echo "  --port         PG 端口 (默认: $PG_PORT)"
    echo "  --user         PG 用户 (默认: $PG_USER)"
    echo "  --password     PG 密码"
    echo "  --database     PG 数据库名 (默认: $PG_DATABASE)"
    echo "  --pg-data-dir  PG 数据目录 (用于 pg_ctl restart)"
    echo "  --output       输出目录 (默认: $OUTPUT_DIR)"
    echo "  --rounds       采集轮数 (默认: $ROUNDS)"
    echo "  --sampling     采样策略: random/lhs (默认: $SAMPLING)"
    echo "  --workload     负载类型: mixed/read_only/high_concurrency/write_heavy/all (默认: all)"
    echo "  --foreground   前台执行（调试用）"
    echo "  -h, --help     显示帮助"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)         PG_HOST="$2"; shift 2 ;;
        --port)         PG_PORT="$2"; shift 2 ;;
        --user)         PG_USER="$2"; shift 2 ;;
        --password)     PG_PASSWORD="$2"; shift 2 ;;
        --database)     PG_DATABASE="$2"; shift 2 ;;
        --pg-data-dir)  PG_DATA_DIR="$2"; shift 2 ;;
        --output)       OUTPUT_DIR="$2"; shift 2 ;;
        --rounds)       ROUNDS="$2"; shift 2 ;;
        --sampling)     SAMPLING="$2"; shift 2 ;;
        --workload)     WORKLOAD="$2"; shift 2 ;;
        --foreground)   FOREGROUND=true; shift ;;
        -h|--help)      usage ;;
        *)              echo "未知参数: $1"; usage ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "  Cost Model 数据采集"
echo "=========================================="
echo "PG 连接:   ${PG_USER}@${PG_HOST}:${PG_PORT}/${PG_DATABASE}"
echo "输出目录:  ${OUTPUT_DIR}"
echo "采集轮数:  ${ROUNDS}"
echo "采样策略:  ${SAMPLING}"
echo ""

# 检查依赖
python3 -c "import psycopg2, yaml" 2>/dev/null || {
    echo "错误: 缺少依赖，请运行: pip install -r requirements.txt"
    exit 1
}

# 检查 PG
if ! pg_isready -q -h $PG_HOST -p $PG_PORT 2>/dev/null; then
    echo "PostgreSQL 未运行，尝试自动启动..."
    if systemctl is-active --quiet postgresql 2>/dev/null; then
        sudo systemctl restart postgresql
    elif command -v pg_ctlcluster &>/dev/null; then
        pg_ctlcluster 16 main start 2>/dev/null
    else
        echo "错误: 无法启动 PostgreSQL"
        exit 1
    fi
    echo "  ✓ PostgreSQL 已启动"
else
    echo "  ✓ PostgreSQL 运行中"
fi

# 检查数据库
echo "检查数据库 ${PG_DATABASE}..."
psql -U $PG_USER -h $PG_HOST -p $PG_PORT -tc \
    "SELECT 1 FROM pg_database WHERE datname = '$PG_DATABASE'" 2>/dev/null | grep -q 1 || {
    echo "  → 创建数据库 ${PG_DATABASE}..."
    psql -U $PG_USER -h $PG_HOST -p $PG_PORT -c "CREATE DATABASE $PG_DATABASE;"
}
echo "  ✓ 数据库 ${PG_DATABASE} 就绪"

# 构建命令
CMD="python3 -u -m cost_model.data.pipeline \
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

# 执行
if [ "$FOREGROUND" = true ]; then
    eval $CMD
else
    LOG_DIR="logs/costmodel"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S).log"
    PID_FILE="${LOG_DIR}/running.pid"

    echo "后台运行..."
    echo "日志: $LOG_FILE"
    nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "PID:  $! (保存到 $PID_FILE)"
    echo ""
    echo "查看进度: tail -f $LOG_FILE"
    echo "停止采集: kill \$(cat $PID_FILE)"
fi
