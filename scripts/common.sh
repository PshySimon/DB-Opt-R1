#!/bin/bash
# 采集脚本公共函数库
# 使用方法: source scripts/common.sh

# ==================== PG 连接默认参数 ====================
PG_HOST="${PG_HOST:-127.0.0.1}"
PG_PORT="${PG_PORT:-5432}"
PG_USER="${PG_USER:-postgres}"
PG_PASSWORD="${PG_PASSWORD:-}"
PG_DATABASE="${PG_DATABASE:-benchmark}"
PG_DATA_DIR="${PG_DATA_DIR:-}"

# ==================== 公共函数 ====================

parse_pg_args() {
    # 解析 PG 连接相关参数，返回剩余参数
    # 用法: eval "$(parse_pg_args "$@")"
    local remaining=()
    while [[ $# -gt 0 ]]; do
        case $1 in
            --host)         PG_HOST="$2"; shift 2 ;;
            --port)         PG_PORT="$2"; shift 2 ;;
            --user)         PG_USER="$2"; shift 2 ;;
            --password)     PG_PASSWORD="$2"; shift 2 ;;
            --database)     PG_DATABASE="$2"; shift 2 ;;
            --pg-data-dir)  PG_DATA_DIR="$2"; shift 2 ;;
            *)              remaining+=("$1"); shift ;;
        esac
    done
    # 把剩余参数写回
    echo "set -- ${remaining[*]}"
}

print_pg_info() {
    echo "PG 连接:   ${PG_USER}@${PG_HOST}:${PG_PORT}/${PG_DATABASE}"
}

check_dependencies() {
    python3 -c "import psycopg2, yaml" 2>/dev/null || {
        echo "错误: 缺少依赖，请运行: pip install -r requirements.txt"
        exit 1
    }
}

check_pg_ready() {
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
}

check_database() {
    echo "检查数据库 ${PG_DATABASE}..."
    psql -U $PG_USER -h $PG_HOST -p $PG_PORT -tc \
        "SELECT 1 FROM pg_database WHERE datname = '$PG_DATABASE'" 2>/dev/null | grep -q 1 || {
        echo "  → 创建数据库 ${PG_DATABASE}..."
        psql -U $PG_USER -h $PG_HOST -p $PG_PORT -c "CREATE DATABASE $PG_DATABASE;"
    }
    echo "  ✓ 数据库 ${PG_DATABASE} 就绪"
}

run_in_background() {
    # 用法: run_in_background <log_dir> <cmd...>
    local LOG_DIR="$1"; shift
    local CMD="$*"

    mkdir -p "$LOG_DIR"
    local LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S).log"
    local PID_FILE="${LOG_DIR}/running.pid"

    echo ""
    echo "命令: $CMD"
    echo "日志: $LOG_FILE"

    nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    local PID=$!
    echo $PID > "$PID_FILE"

    echo "PID:  $PID (已保存到 $PID_FILE)"
    echo ""
    echo "查看进度: tail -f $LOG_FILE"
    echo "停止采集: kill \$(cat $PID_FILE)"
}

print_pg_help() {
    echo "  --host         PG 主机地址 (默认: $PG_HOST)"
    echo "  --port         PG 端口 (默认: $PG_PORT)"
    echo "  --user         PG 用户 (默认: $PG_USER)"
    echo "  --password     PG 密码"
    echo "  --database     PG 数据库名 (默认: $PG_DATABASE)"
    echo "  --pg-data-dir  PG 数据目录"
}
