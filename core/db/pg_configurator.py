"""
PG 配置器：应用 knob 配置到 PostgreSQL 并重启
"""

import subprocess
import time
import logging

logger = logging.getLogger(__name__)

# 默认超时
RESTART_TIMEOUT = 30  # 秒
READY_CHECK_INTERVAL = 1  # 秒


class PGConfigurator:
    """管理 PostgreSQL 配置的应用和重启"""

    def __init__(self, pg_host: str = "127.0.0.1", pg_port: int = 5432,
                 pg_user: str = "postgres", pg_password: str = "",
                 pg_database: str = "postgres", pg_data_dir: str = None):
        self.pg_host = pg_host
        self.pg_port = pg_port
        self.pg_user = pg_user
        self.pg_password = pg_password
        self.pg_database = pg_database
        self.pg_data_dir = pg_data_dir

    def apply(self, knob_config: dict, needs_restart: bool = False):
        """应用 knob 配置

        Args:
            knob_config: {knob_name: value} 字典
            needs_restart: 是否需要重启（有 static knob 变更时为 True）
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        for name, value in knob_config.items():
            try:
                cursor.execute(f"ALTER SYSTEM SET {name} = %s", (str(value),))
                logger.debug(f"SET {name} = {value}")
            except Exception as e:
                logger.warning(f"设置 {name} = {value} 失败: {e}")
                conn.rollback()

        conn.close()

        if needs_restart:
            self.restart()
        else:
            self.reload()

    def reload(self):
        """重载配置（不重启，仅对 dynamic knob 生效）"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT pg_reload_conf()")
            conn.close()
            logger.info("PG 配置已 reload")
        except Exception as e:
            logger.error(f"reload 失败: {e}")
            raise

    PG_LOG_PATH = "/var/log/postgresql/postgresql-16-main.log"

    def restart(self):
        """重启 PostgreSQL（重启前清空日志）"""
        logger.info("正在重启 PostgreSQL...")

        # 清空日志文件，确保每个场景只采集自己的日志
        try:
            subprocess.run(
                ["sudo", "truncate", "-s", "0", self.PG_LOG_PATH],
                capture_output=True, timeout=5
            )
        except Exception:
            pass

        self._do_restart()

        # 等待 PG 就绪
        try:
            self._wait_ready()
        except TimeoutError:
            logger.warning("重启超时，尝试 force_reset 后再重启")
            self.force_reset()
            self._do_restart()
            self._wait_ready()  # 再次超时就真的抛异常

        logger.info("PostgreSQL 已重启")

    def _do_restart(self):
        """执行实际重启命令"""
        if self.pg_data_dir:
            self._run_cmd(f"pg_ctl -D {self.pg_data_dir} restart -w -t {RESTART_TIMEOUT}")
        else:
            import shutil
            if self._is_systemd_managed():
                self._run_cmd("sudo systemctl restart postgresql")
            elif shutil.which("pg_ctlcluster"):
                self._run_cmd("pg_ctlcluster 16 main restart")
            else:
                self._run_cmd("sudo systemctl restart postgresql")

    def safe_restart(self):
        """安全重启：先 force_reset 清配置，再重启（PG 挂掉时的恢复入口）"""
        logger.info("执行安全重启：force_reset → restart")
        self.force_reset()
        self._do_restart()
        self._wait_ready()
        logger.info("安全重启完成")

    def reset_to_default(self):
        """重置所有 knob 到默认值"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("ALTER SYSTEM RESET ALL")
            conn.close()
            logger.info("已重置所有 knob 到默认值")
        except Exception as e:
            logger.warning(f"SQL 重置失败: {e}，尝试直接清除配置文件")
            self.force_reset()

    def force_reset(self):
        """强制重置：直接清空 postgresql.auto.conf（PG 挂掉时用）"""
        import glob
        auto_conf_patterns = [
            "/var/lib/postgresql/*/main/postgresql.auto.conf",
            f"{self.pg_data_dir}/postgresql.auto.conf" if self.pg_data_dir else "",
        ]
        for pattern in auto_conf_patterns:
            if not pattern:
                continue
            for path in glob.glob(pattern):
                try:
                    # 用 sudo truncate，因为文件属于 postgres 用户
                    result = subprocess.run(
                        ['sudo', 'truncate', '-s', '0', path],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode != 0:
                        logger.error(f"truncate {path} 失败: {result.stderr}")
                    else:
                        logger.info(f"已清空 {path}")
                except Exception as e:
                    logger.error(f"清空 {path} 失败: {e}")

    def verify_config(self, knob_config: dict) -> dict:
        """验证配置是否已生效，返回实际值"""
        conn = self._get_connection()
        cursor = conn.cursor()
        actual = {}
        for name in knob_config:
            try:
                cursor.execute(f"SHOW {name}")
                actual[name] = cursor.fetchone()[0]
            except Exception:
                actual[name] = None
        conn.close()
        return actual

    def _get_connection(self):
        import psycopg2
        conn = psycopg2.connect(
            host=self.pg_host, port=self.pg_port,
            user=self.pg_user, password=self.pg_password,
            database=self.pg_database
        )
        conn.autocommit = True  # ALTER SYSTEM 不能在事务里执行
        return conn

    def _wait_ready(self):
        """轮询等待 PG 就绪"""
        for i in range(RESTART_TIMEOUT):
            try:
                result = subprocess.run(
                    ["pg_isready", "-h", self.pg_host, "-p", str(self.pg_port)],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    return
            except Exception:
                pass
            time.sleep(READY_CHECK_INTERVAL)

        raise TimeoutError(f"PostgreSQL 在 {RESTART_TIMEOUT}s 内未就绪")

    def _is_systemd_managed(self) -> bool:
        """检测 PostgreSQL 是否由 systemd 管理"""
        try:
            result = subprocess.run(
                ["systemctl", "is-active", "postgresql"],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _run_cmd(self, cmd: str):
        """执行 shell 命令"""
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=RESTART_TIMEOUT
        )
        if result.returncode != 0:
            logger.error(f"命令失败: {cmd}\n{result.stderr}")
            raise RuntimeError(f"命令执行失败: {cmd}")
