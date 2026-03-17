#!/bin/bash
# db-opt-r1 环境准备脚本（Ubuntu）
# 安装 PostgreSQL 16、pgbench、Python 依赖

set -e

PG_VERSION="${PG_VERSION:-16}"
PG_DATABASE="${PG_DATABASE:-benchmark}"
PG_USER="${PG_USER:-postgres}"

echo "=========================================="
echo "  db-opt-r1 环境准备"
echo "  PostgreSQL $PG_VERSION / Ubuntu"
echo "=========================================="

# ==================== 1. 系统依赖 ====================
echo ""
echo "[1/5] 安装系统依赖..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    wget curl gnupg2 lsb-release \
    python3 python3-pip python3-venv \
    sysstat linux-tools-common

# ==================== 2. PostgreSQL ====================
echo ""
echo "[2/5] 安装 PostgreSQL $PG_VERSION..."

# 添加官方 APT 源
if ! apt-cache policy postgresql-$PG_VERSION 2>/dev/null | grep -q "Candidate"; then
    sudo sh -c "echo 'deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main' > /etc/apt/sources.list.d/pgdg.list"
    wget -qO- https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
    sudo apt-get update -qq
fi

sudo apt-get install -y -qq \
    postgresql-$PG_VERSION \
    postgresql-contrib-$PG_VERSION

# 确认安装
pg_config --version
echo "pgbench 路径: $(which pgbench)"

# ==================== 3. 配置 PostgreSQL ====================
echo ""
echo "[3/5] 配置 PostgreSQL..."

PG_CONF="/etc/postgresql/$PG_VERSION/main/postgresql.conf"
PG_HBA="/etc/postgresql/$PG_VERSION/main/pg_hba.conf"
PG_DATA="/var/lib/postgresql/$PG_VERSION/main"

# 开启统计收集（工具需要）
sudo sed -i "s/#track_activities = on/track_activities = on/" $PG_CONF
sudo sed -i "s/#track_counts = on/track_counts = on/" $PG_CONF
sudo sed -i "s/#track_io_timing = off/track_io_timing = on/" $PG_CONF

# 日志配置（get_recent_logs 需要）
sudo sed -i "s/#log_min_duration_statement = -1/log_min_duration_statement = 1000/" $PG_CONF
sudo sed -i "s/#log_temp_files = -1/log_temp_files = 0/" $PG_CONF
sudo sed -i "s/#log_lock_waits = off/log_lock_waits = on/" $PG_CONF
sudo sed -i "s/#log_checkpoints = off/log_checkpoints = on/" $PG_CONF

# 允许本地密码登录
sudo sed -i "s/local\s\+all\s\+all\s\+peer/local   all             all                                     trust/" $PG_HBA

# 修改了配置，需要重启使之生效（apt install 已自动启动了服务）
sudo systemctl restart postgresql
sudo systemctl enable postgresql

# 验证服务状态
pg_isready -q && echo "PostgreSQL $PG_VERSION 运行正常" || { echo "PostgreSQL 启动失败！"; exit 1; }

# ==================== 4. 创建 benchmark 数据库 ====================
echo ""
echo "[4/5] 创建 benchmark 数据库..."

sudo -u postgres psql -c "CREATE DATABASE $PG_DATABASE;" 2>/dev/null || echo "数据库 $PG_DATABASE 已存在"
sudo -u postgres psql -c "ALTER USER $PG_USER WITH PASSWORD 'postgres';" 2>/dev/null

echo "数据库: $PG_DATABASE"
echo "用户:   $PG_USER / postgres"

# ==================== 5. Python 依赖 ====================
echo ""
echo "[5/5] 安装 Python 依赖..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

pip3 install -r "$PROJECT_DIR/requirements.txt"

# ==================== 完成 ====================
echo ""
echo "=========================================="
echo "  环境准备完成！"
echo "=========================================="
echo ""
echo "PostgreSQL: $PG_VERSION"
echo "数据目录:   $PG_DATA"
echo "数据库:     $PG_DATABASE"
echo "连接:       psql -U $PG_USER -d $PG_DATABASE"
echo ""
echo "下一步:"
echo "  1. 初始化 benchmark 数据:"
echo "     bash scripts/collect_data.sh --init --database $PG_DATABASE"
echo "  2. 开始采集:"
echo "     bash scripts/collect_data.sh --rounds 100 --database $PG_DATABASE"
