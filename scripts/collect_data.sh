#!/bin/bash
# db-opt-r1 数据采集脚本

set -e

# 默认参数
PG_HOST="${PG_HOST:-127.0.0.1}"
PG_PORT="${PG_PORT:-5432}"
PG_USER="${PG_USER:-postgres}"
PG_PASSWORD="${PG_PASSWORD:-}"
PG_DATABASE="${PG_DATABASE:-postgres}"
PG_DATA_DIR="${PG_DATA_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./cost_model/data/raw}"
ROUNDS="${ROUNDS:-100}"
SAMPLING="${SAMPLING:-random}"
INIT_BENCHMARK=false

usage() {
    echo "用法: $0 [选项]"
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
    echo "  --init         初始化 benchmark 数据"
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
        --init)         INIT_BENCHMARK=true; shift ;;
        -h|--help)      usage ;;
        *)              echo "未知参数: $1"; usage ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "  db-opt-r1 数据采集 Pipeline"
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

# 自动创建数据库（如果不存在）
echo "检查数据库 ${PG_DATABASE}..."
psql -U $PG_USER -h $PG_HOST -p $PG_PORT -tc "SELECT 1 FROM pg_database WHERE datname = '$PG_DATABASE'" | grep -q 1 || {
    echo "  → 创建数据库 ${PG_DATABASE}..."
    psql -U $PG_USER -h $PG_HOST -p $PG_PORT -c "CREATE DATABASE $PG_DATABASE;"
}
echo "  ✓ 数据库 ${PG_DATABASE} 就绪"

# 构建命令
CMD="python -m cost_model.data.pipeline \
    --config configs/knob_space.yaml \
    --host $PG_HOST --port $PG_PORT \
    --user $PG_USER --password $PG_PASSWORD \
    --database $PG_DATABASE \
    --output $OUTPUT_DIR \
    --rounds $ROUNDS \
    --sampling $SAMPLING"

if [ -n "$PG_DATA_DIR" ]; then
    CMD="$CMD --pg-data-dir $PG_DATA_DIR"
fi

if [ "$INIT_BENCHMARK" = true ]; then
    CMD="$CMD --init"
fi

eval $CMD
