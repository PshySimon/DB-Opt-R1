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
# 清理可能残留的失效 PG 源
if [ -f /etc/apt/sources.list.d/pgdg.list ]; then
    echo "  → 清理旧的 pgdg 源文件..."
    $SUDO rm -f /etc/apt/sources.list.d/pgdg.list
fi
echo "  → apt-get update..."
$SUDO apt-get update
echo "  → 安装 wget, curl, gnupg2, lsb-release, python3, sysstat..."
$SUDO apt-get install -y \
    wget curl gnupg2 lsb-release \
    python3 python3-pip python3-venv \
    sysstat 2>/dev/null || true
echo "  ✓ 系统依赖安装完成"

# ==================== 2. PostgreSQL ====================
echo ""
echo "[2/5] 安装 PostgreSQL $PG_VERSION..."

# 安装 PostgreSQL（优先系统源，无需额外添加源）
echo "  → 尝试从系统源安装 postgresql-$PG_VERSION..."
if ! $SUDO apt-get install -y postgresql-$PG_VERSION postgresql-contrib-$PG_VERSION 2>/dev/null; then
    echo "  → 系统源无 PG $PG_VERSION，尝试添加官方 PG 源..."
    $SUDO sh -c "echo 'deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main' > /etc/apt/sources.list.d/pgdg.list"
    echo "  → 导入 PG GPG Key..."
    wget -qO- https://www.postgresql.org/media/keys/ACCC4CF8.asc | $SUDO apt-key add - 2>/dev/null
    echo "  → apt-get update..."
    $SUDO apt-get update
    echo "  → 安装 postgresql-$PG_VERSION..."
    $SUDO apt-get install -y postgresql-$PG_VERSION postgresql-contrib-$PG_VERSION
fi
echo "  ✓ PostgreSQL 安装完成"

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
    echo "  ⚠ 未找到 postgresql.conf，跳过配置修改"
else
    echo "  → 配置文件: $PG_CONF"
    echo "  → HBA 文件: $PG_HBA"
    echo "  → 数据目录: $PG_DATA"
    echo "  → 开启 track_activities, track_counts, track_io_timing"
    # 开启统计收集
    $SUDO sed -i "s/#track_activities = on/track_activities = on/" $PG_CONF
    $SUDO sed -i "s/#track_counts = on/track_counts = on/" $PG_CONF
    $SUDO sed -i "s/#track_io_timing = off/track_io_timing = on/" $PG_CONF

    echo "  → 配置慢查询日志、临时文件日志、锁等待日志"
    # 日志配置
    $SUDO sed -i "s/#log_min_duration_statement = -1/log_min_duration_statement = 1000/" $PG_CONF
    $SUDO sed -i "s/#log_temp_files = -1/log_temp_files = 0/" $PG_CONF
    $SUDO sed -i "s/#log_lock_waits = off/log_lock_waits = on/" $PG_CONF
    $SUDO sed -i "s/#log_checkpoints = off/log_checkpoints = on/" $PG_CONF

    echo "  → 配置本地免密登录"
    # 允许本地免密登录（Unix socket + TCP 127.0.0.1 都改成 trust）
    if [ -n "$PG_HBA" ]; then
        $SUDO sed -i 's/local\s\+all\s\+postgres\s\+peer/local   all             postgres                                trust/' $PG_HBA
        $SUDO sed -i 's/local\s\+all\s\+all\s\+peer/local   all             all                                     trust/' $PG_HBA
        $SUDO sed -i 's|host\s\+all\s\+all\s\+127.0.0.1/32\s\+scram-sha-256|host    all             all             127.0.0.1/32            trust|' $PG_HBA
        $SUDO sed -i 's|host\s\+all\s\+all\s\+::1/128\s\+scram-sha-256|host    all             all             ::1/128                 trust|' $PG_HBA
    fi
    echo "  ✓ 配置修改完成"
fi

# 启动 PostgreSQL（兼容 systemd 和 pg_ctl）
echo "  → 启动 PostgreSQL..."
if command -v systemctl &>/dev/null && systemctl is-system-running &>/dev/null 2>&1; then
    echo "  → systemd 模式"
    $SUDO systemctl restart postgresql
    $SUDO systemctl enable postgresql
else
    echo "  → 容器模式（pg_ctlcluster）"
    $SUDO mkdir -p /var/log/postgresql
    pg_ctlcluster $PG_VERSION main stop 2>/dev/null || true
    pg_ctlcluster $PG_VERSION main start
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

# 创建数据库
echo "  → 创建数据库 $PG_DATABASE..."
if $SUDO -u postgres psql -c "CREATE DATABASE $PG_DATABASE;"; then
    echo "  ✓ 数据库 $PG_DATABASE 创建成功"
else
    echo "  ⚠ 数据库 $PG_DATABASE 已存在或创建失败（见上方输出）"
fi

# 设置密码
echo "  → 设置用户 $PG_USER 密码..."
if $SUDO -u postgres psql -c "ALTER USER $PG_USER WITH PASSWORD 'postgres';"; then
    echo "  ✓ 密码设置成功"
else
    echo "  ❌ 密码设置失败（见上方输出）"
fi

echo "  数据库: $PG_DATABASE"
echo "  用户:   $PG_USER / postgres"

# 初始化 pgbench 数据（仅在未初始化时执行）
echo "  → 检查 pgbench 是否已初始化..."
if $SUDO -u postgres psql -d $PG_DATABASE -tc "SELECT 1 FROM pg_tables WHERE tablename = 'pgbench_accounts'" | grep -q 1; then
    echo "  ✓ pgbench 已初始化，跳过"
else
    echo "  → 初始化 pgbench (scale factor = 10)..."
    if $SUDO -u postgres pgbench -i -s 10 $PG_DATABASE; then
        echo "  ✓ pgbench 初始化完成"
    else
        echo "  ❌ pgbench 初始化失败"
    fi
fi

# ==================== 5. Python 依赖 ====================
echo ""
echo "[5/5] 安装 Python 依赖..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    echo "  → pip install -r requirements.txt..."
    pip3 install -r "$PROJECT_DIR/requirements.txt"
    echo "  ✓ Python 依赖安装完成"
else
    echo "  ⚠ 未找到 requirements.txt，跳过"
fi

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
