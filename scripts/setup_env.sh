#!/bin/bash
# db-opt-r1 环境准备脚本
# 支持两种模式：系统安装（有 sudo）/ 容器模式（无 sudo，已是 root）

set -e

PG_VERSION="${PG_VERSION:-16}"
PG_DATABASE="${PG_DATABASE:-benchmark}"
PG_USER="${PG_USER:-postgres}"

# 自动检测是否需要 sudo
if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
    echo "检测到 root 用户，无需 sudo"
else
    SUDO="sudo"
    echo "非 root 用户，使用 sudo"
fi

echo "=========================================="
echo "  db-opt-r1 环境准备"
echo "  PostgreSQL $PG_VERSION"
echo "=========================================="

# ==================== 1. 系统依赖 ====================
echo ""
echo "[1/5] 安装系统依赖..."
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq \
    wget curl gnupg2 lsb-release \
    python3 python3-pip python3-venv \
    sysstat 2>/dev/null || true

# ==================== 2. PostgreSQL ====================
echo ""
echo "[2/5] 安装 PostgreSQL $PG_VERSION..."

# 添加官方 APT 源
if ! dpkg -l postgresql-$PG_VERSION 2>/dev/null | grep -q "^ii"; then
    $SUDO sh -c "echo 'deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main' > /etc/apt/sources.list.d/pgdg.list"
    wget -qO- https://www.postgresql.org/media/keys/ACCC4CF8.asc | $SUDO apt-key add - 2>/dev/null
    $SUDO apt-get update -qq
fi

$SUDO apt-get install -y -qq \
    postgresql-$PG_VERSION \
    postgresql-contrib-$PG_VERSION

# 确认安装
pg_config --version
echo "pgbench 路径: $(which pgbench)"

# ==================== 3. 配置 PostgreSQL ====================
echo ""
echo "[3/5] 配置 PostgreSQL..."

# 检测配置文件和数据目录
PG_CONF="/etc/postgresql/$PG_VERSION/main/postgresql.conf"
PG_HBA="/etc/postgresql/$PG_VERSION/main/pg_hba.conf"
PG_DATA="/var/lib/postgresql/$PG_VERSION/main"

if [ ! -f "$PG_CONF" ]; then
    # 容器环境可能路径不同
    PG_CONF=$(find /etc/postgresql -name "postgresql.conf" 2>/dev/null | head -1)
    PG_HBA=$(find /etc/postgresql -name "pg_hba.conf" 2>/dev/null | head -1)
    PG_DATA=$(find /var/lib/postgresql -name "main" -type d 2>/dev/null | head -1)
fi

if [ -z "$PG_CONF" ]; then
    echo "警告: 未找到 postgresql.conf，跳过配置修改"
else
    # 开启统计收集
    $SUDO sed -i "s/#track_activities = on/track_activities = on/" $PG_CONF
    $SUDO sed -i "s/#track_counts = on/track_counts = on/" $PG_CONF
    $SUDO sed -i "s/#track_io_timing = off/track_io_timing = on/" $PG_CONF

    # 日志配置
    $SUDO sed -i "s/#log_min_duration_statement = -1/log_min_duration_statement = 1000/" $PG_CONF
    $SUDO sed -i "s/#log_temp_files = -1/log_temp_files = 0/" $PG_CONF
    $SUDO sed -i "s/#log_lock_waits = off/log_lock_waits = on/" $PG_CONF
    $SUDO sed -i "s/#log_checkpoints = off/log_checkpoints = on/" $PG_CONF

    # 允许本地免密登录
    if [ -n "$PG_HBA" ]; then
        $SUDO sed -i "s/local\s\+all\s\+all\s\+peer/local   all             all                                     trust/" $PG_HBA
    fi
fi

# 启动 PostgreSQL（兼容 systemd 和 pg_ctl）
echo "启动 PostgreSQL..."
if command -v systemctl &>/dev/null && systemctl is-system-running &>/dev/null 2>&1; then
    # 有 systemd（物理机/VM）
    $SUDO systemctl restart postgresql
    $SUDO systemctl enable postgresql
else
    # 无 systemd（容器）
    echo "容器模式：使用 pg_ctl 启动"
    # 先停现有进程
    su - postgres -c "pg_ctl -D $PG_DATA stop" 2>/dev/null || true
    su - postgres -c "pg_ctl -D $PG_DATA start -l /var/log/postgresql/startup.log"
fi

# 等待就绪
for i in $(seq 1 10); do
    if pg_isready -q; then
        echo "PostgreSQL $PG_VERSION 运行正常"
        break
    fi
    echo "  等待 PostgreSQL 启动... ($i/10)"
    sleep 1
done

pg_isready -q || { echo "PostgreSQL 启动失败！"; exit 1; }

# ==================== 4. 创建 benchmark 数据库 ====================
echo ""
echo "[4/5] 创建 benchmark 数据库..."

# 容器里已经是 root，尝试直接用 postgres 用户执行
su - postgres -c "psql -c \"CREATE DATABASE $PG_DATABASE;\"" 2>/dev/null || echo "数据库 $PG_DATABASE 已存在"
su - postgres -c "psql -c \"ALTER USER $PG_USER WITH PASSWORD 'postgres';\"" 2>/dev/null

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
