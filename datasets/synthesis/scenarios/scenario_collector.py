"""
场景采集器：扩展 DataCollector，新增系统级指标、等待事件、慢查询、日志采集。

用于在真机上采集完整的故障场景数据。
"""

import logging
import subprocess

from core.db.collector import DataCollector

logger = logging.getLogger(__name__)


class ScenarioCollector(DataCollector):
    """扩展 DataCollector，额外采集系统级指标"""

    def collect_scenario(self) -> dict:
        """采集完整场景数据（基础快照 + 系统指标 + 等待事件 + 慢查询 + 日志）"""
        snapshot = self.collect_snapshot()

        # 新增采集
        snapshot["system"] = self._collect_system_stats()
        snapshot["wait_events"] = self._collect_wait_events()
        snapshot["slow_queries"] = self._collect_slow_queries()
        snapshot["logs"] = self._collect_recent_logs()

        return snapshot

    def _collect_system_stats(self) -> dict:
        """采集 CPU / 内存 / IO / swap 利用率"""
        stats = {}

        # CPU 使用率（从 /proc/stat 读）
        try:
            output = subprocess.check_output(
                "top -bn1 | grep 'Cpu(s)' | awk '{print $2}'",
                shell=True, text=True, timeout=5
            )
            stats["cpu_usage_pct"] = round(float(output.strip()), 1)
        except Exception:
            stats["cpu_usage_pct"] = None

        # 内存使用率
        try:
            output = subprocess.check_output(
                "free -m | grep Mem | awk '{printf \"%.1f\", $3/$2*100}'",
                shell=True, text=True, timeout=5
            )
            stats["memory_usage_pct"] = float(output.strip())
        except Exception:
            stats["memory_usage_pct"] = None

        # Swap
        try:
            output = subprocess.check_output(
                "free -m | grep Swap | awk '{print $3}'",
                shell=True, text=True, timeout=5
            )
            stats["swap_usage_mb"] = int(output.strip() or 0)
        except Exception:
            stats["swap_usage_mb"] = 0

        # IO（iostat）
        try:
            output = subprocess.check_output(
                "iostat -dx 1 1 | tail -n +4 | head -1 | awk '{print $4, $5, $NF}'",
                shell=True, text=True, timeout=10
            )
            parts = output.strip().split()
            if len(parts) >= 3:
                stats["disk_read_iops"] = float(parts[0])
                stats["disk_write_iops"] = float(parts[1])
                stats["disk_io_util_pct"] = float(parts[2])
        except Exception:
            pass

        return {k: v for k, v in stats.items() if v is not None}

    def _collect_wait_events(self) -> list:
        """采集等待事件分布（pg_stat_activity）"""
        try:
            cursor = self.pg_conn.cursor()
            cursor.execute("""
                SELECT wait_event_type || ':' || wait_event AS event,
                       count(*) AS cnt
                FROM pg_stat_activity
                WHERE wait_event IS NOT NULL AND state = 'active'
                GROUP BY 1
                ORDER BY 2 DESC
                LIMIT 10
            """)
            total = 0
            rows = []
            for event, cnt in cursor.fetchall():
                rows.append({"event": event, "count": int(cnt)})
                total += cnt
            cursor.close()

            # 转为百分比
            if total > 0:
                for r in rows:
                    r["pct"] = round(r["count"] / total * 100, 1)
                    del r["count"]

            return rows
        except Exception as e:
            logger.warning(f"采集等待事件失败: {e}")
            return []

    def _collect_slow_queries(self) -> list:
        """采集慢查询 Top 5（pg_stat_statements）"""
        try:
            cursor = self.pg_conn.cursor()
            cursor.execute("""
                SELECT query, round(mean_exec_time::numeric, 2) AS avg_time_ms,
                       calls
                FROM pg_stat_statements
                WHERE calls > 0
                ORDER BY mean_exec_time DESC
                LIMIT 5
            """)
            rows = []
            for query, avg_time, calls in cursor.fetchall():
                # 截断过长的 query
                q = str(query)[:200] + ("..." if len(str(query)) > 200 else "")
                rows.append({
                    "query": q,
                    "avg_time_ms": float(avg_time),
                    "calls": int(calls),
                })
            cursor.close()
            return rows
        except Exception as e:
            logger.warning(f"采集慢查询失败（可能未启用 pg_stat_statements）: {e}")
            return []

    def _collect_recent_logs(self, log_path: str = None, n: int = 20) -> list:
        """读取 PG 最近的 WARNING/ERROR 日志"""
        if log_path is None:
            log_path = "/var/log/postgresql/postgresql-16-main.log"

        try:
            result = subprocess.run(
                ["sudo", "tail", "-n", "500", log_path],
                capture_output=True, text=True, timeout=5
            )
            logs = []
            for line in result.stdout.strip().split("\n"):
                for level in ("WARNING", "ERROR", "FATAL"):
                    if level in line:
                        logs.append({"level": level, "message": line.strip()[-200:]})
                        break
            return logs[-n:]
        except Exception as e:
            logger.warning(f"读取日志失败: {e}")
            return []
