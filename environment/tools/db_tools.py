"""
DB 调优工具集

每个工具类包含 execute_real() 和 execute_simulated() 两个方法，
execute() 根据 mode 自动分发。
"""

import os
import json
import time
import subprocess
import logging

from core.tool.tool_base import Tool

logger = logging.getLogger(__name__)


class DBTool(Tool):
    """DB 工具基类，扩展 real/simulated 双模式"""

    def __init__(self, name, description, parameters,
                 mode="train", config=None, env_state=None, **kwargs):
        super().__init__(name, description, parameters)
        self.mode = mode
        self.config = config
        self.env_state = env_state

    def execute(self, args):
        if self.mode == "real":
            return self.execute_real(args)
        else:
            return self.execute_simulated(args)

    def execute_real(self, args) -> str:
        raise NotImplementedError

    def execute_simulated(self, args) -> str:
        raise NotImplementedError

    def _get_conn(self):
        return self.config.get_db_connection()


# ==================== 观察类 ====================

class GetHardwareInfoTool(DBTool):
    def __init__(self, **kwargs):
        super().__init__(
            name="get_hardware_info",
            description="获取硬件环境信息（CPU、内存、磁盘等）",
            parameters={"type": "object", "properties": {}, "required": []},
            **kwargs
        )

    def execute_real(self, args):
        info = {
            "cpu_count": os.cpu_count(),
            "cpu_model": self._cmd("grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2").strip() or "unknown",
            "total_memory_gb": round(
                int(self._cmd("grep MemTotal /proc/meminfo | awk '{print $2}'").strip() or 0) / 1024 / 1024, 1
            ),
            "disk_type": "SSD" if self._cmd("lsblk -d -o rota | grep -v ROTA | head -1").strip() == "0" else "HDD",
        }
        return json.dumps(info, ensure_ascii=False, indent=2)

    def execute_simulated(self, args):
        info = {k.replace("hw_", ""): v for k, v in self.env_state.items() if k.startswith("hw_")}
        return json.dumps(info, ensure_ascii=False, indent=2)

    def _cmd(self, cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True, timeout=5)
        except Exception:
            return ""


class GetCurrentConfigTool(DBTool):
    def __init__(self, **kwargs):
        super().__init__(
            name="get_current_config",
            description="获取当前 PostgreSQL 的 knob 配置",
            parameters={
                "type": "object",
                "properties": {
                    "knob_names": {"type": "string", "description": "要查询的 knob 名称，逗号分隔。留空返回所有"}
                },
                "required": []
            },
            **kwargs
        )

    def execute_real(self, args):
        conn = self._get_conn()
        cursor = conn.cursor()
        names = args.get("knob_names", "")
        if names:
            config = {}
            for name in [n.strip() for n in names.split(",")]:
                try:
                    cursor.execute(f"SHOW {name}")
                    config[name] = cursor.fetchone()[0]
                except Exception:
                    config[name] = "unknown"
        else:
            cursor.execute("SHOW ALL")
            config = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.close()
        conn.close()
        return json.dumps(config, ensure_ascii=False, indent=2)

    def execute_simulated(self, args):
        names = args.get("knob_names", "")
        knobs = {k.replace("knob_", ""): v for k, v in self.env_state.items() if k.startswith("knob_")}
        if names:
            name_list = [n.strip() for n in names.split(",")]
            knobs = {k: knobs.get(k, "unknown") for k in name_list}
        return json.dumps(knobs, ensure_ascii=False, indent=2)


class GetDBMetricsTool(DBTool):
    def __init__(self, **kwargs):
        super().__init__(
            name="get_db_metrics",
            description="获取 DB 运行时指标（缓冲区命中率、临时文件数、死元组比例等）",
            parameters={"type": "object", "properties": {}, "required": []},
            **kwargs
        )

    def execute_real(self, args):
        conn = self._get_conn()
        cursor = conn.cursor()
        metrics = {}
        try:
            cursor.execute("SELECT sum(blks_hit), sum(blks_read) FROM pg_stat_database")
            hits, reads = [float(x or 0) for x in cursor.fetchone()]
            metrics["buffer_hit_rate"] = round(hits / (hits + reads), 4) if (hits + reads) > 0 else 0
        except Exception:
            pass
        try:
            cursor.execute("SELECT sum(temp_files), sum(temp_bytes) FROM pg_stat_database")
            row = cursor.fetchone()
            metrics["temp_files_count"] = int(row[0] or 0)
            metrics["temp_bytes_mb"] = round(float(row[1] or 0) / 1024 / 1024, 2)
        except Exception:
            pass
        try:
            cursor.execute("SELECT sum(n_live_tup), sum(n_dead_tup) FROM pg_stat_user_tables")
            live, dead = [float(x or 0) for x in cursor.fetchone()]
            metrics["dead_tuple_ratio"] = round(dead / (live + dead), 4) if (live + dead) > 0 else 0
        except Exception:
            pass
        try:
            cursor.execute("SELECT sum(seq_scan), sum(idx_scan) FROM pg_stat_user_tables")
            seq, idx = [float(x or 0) for x in cursor.fetchone()]
            metrics["seq_scan_ratio"] = round(seq / (seq + idx), 4) if (seq + idx) > 0 else 0
        except Exception:
            pass
        try:
            cursor.execute("SELECT sum(deadlocks) FROM pg_stat_database")
            metrics["deadlocks"] = int(cursor.fetchone()[0] or 0)
        except Exception:
            pass
        cursor.close()
        conn.close()
        return json.dumps(metrics, ensure_ascii=False, indent=2)

    def execute_simulated(self, args):
        metrics = {k.replace("metric_", ""): v for k, v in self.env_state.items() if k.startswith("metric_")}
        return json.dumps(metrics, ensure_ascii=False, indent=2)


class GetWorkloadInfoTool(DBTool):
    def __init__(self, **kwargs):
        super().__init__(
            name="get_workload_info",
            description="获取 workload 特征（数据库大小、活跃连接数、表数量等）",
            parameters={"type": "object", "properties": {}, "required": []},
            **kwargs
        )

    def execute_real(self, args):
        conn = self._get_conn()
        cursor = conn.cursor()
        info = {}
        try:
            cursor.execute("SELECT pg_database_size(current_database())")
            info["database_size_gb"] = round(cursor.fetchone()[0] / 1024**3, 2)
        except Exception:
            pass
        try:
            cursor.execute("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
            info["active_connections"] = cursor.fetchone()[0]
        except Exception:
            pass
        try:
            cursor.execute("SELECT count(*) FROM pg_stat_user_tables")
            info["table_count"] = cursor.fetchone()[0]
        except Exception:
            pass
        cursor.close()
        conn.close()
        return json.dumps(info, ensure_ascii=False, indent=2)

    def execute_simulated(self, args):
        info = {k.replace("sw_", ""): v for k, v in self.env_state.items() if k.startswith("sw_")}
        return json.dumps(info, ensure_ascii=False, indent=2)


class GetRecentLogsTool(DBTool):
    def __init__(self, **kwargs):
        super().__init__(
            name="get_recent_logs",
            description="获取最近的 PostgreSQL 日志（慢查询、报错等）",
            parameters={
                "type": "object",
                "properties": {
                    "lines": {"type": "integer", "description": "返回行数，默认 50"},
                    "level": {"type": "string", "description": "最低日志级别，默认 WARNING"}
                },
                "required": []
            },
            **kwargs
        )

    def execute_real(self, args):
        lines = args.get("lines", 50)
        level = args.get("level", "WARNING")
        log_path = self.config.get("tools.database.log_path",
                                   "/var/log/postgresql/postgresql-16-main.log")
        try:
            result = subprocess.run(["tail", "-n", str(lines * 3), log_path],
                                    capture_output=True, text=True, timeout=5)
            level_pri = {"DEBUG": 0, "LOG": 1, "INFO": 2, "NOTICE": 3,
                         "WARNING": 4, "ERROR": 5, "FATAL": 6}
            min_pri = level_pri.get(level.upper(), 4)
            filtered = [l for l in result.stdout.strip().split("\n")
                        if any(lvl in l and pri >= min_pri for lvl, pri in level_pri.items())]
            return "\n".join(filtered[-lines:]) or "No matching log entries."
        except Exception as e:
            return f"Error reading logs: {e}"

    def execute_simulated(self, args):
        logs = []
        for k, v in self.env_state.items():
            if k.startswith("metric_") and isinstance(v, (int, float)):
                if "buffer_hit" in k and v < 0.9:
                    logs.append("WARNING: buffer hit rate low, consider increasing shared_buffers")
                if "temp_files" in k and v > 100:
                    logs.append(f"WARNING: {int(v)} temporary files, consider increasing work_mem")
                if "dead_tuple" in k and v > 0.1:
                    logs.append("WARNING: high dead tuple ratio, check autovacuum settings")
        return "\n".join(logs) if logs else "No warnings detected."


# ==================== 行动类 ====================

class SetKnobTool(DBTool):
    def __init__(self, **kwargs):
        super().__init__(
            name="set_knob",
            description="设置 PostgreSQL 参数",
            parameters={
                "type": "object",
                "properties": {
                    "knobs": {"type": "string", "description": 'JSON 格式的 knob 配置，如 {"shared_buffers": "8GB"}'}
                },
                "required": ["knobs"]
            },
            **kwargs
        )

    def execute_real(self, args):
        knobs = json.loads(args["knobs"])
        conn = self._get_conn()
        cursor = conn.cursor()
        success, pending_restart, failed = [], [], []
        for name, value in knobs.items():
            try:
                cursor.execute(f"ALTER SYSTEM SET {name} = %s", (str(value),))
                cursor.execute("SELECT context FROM pg_settings WHERE name = %s", (name,))
                row = cursor.fetchone()
                (pending_restart if row and row[0] == "postmaster" else success).append(name)
            except Exception as e:
                failed.append({"name": name, "error": str(e)})
                conn.rollback()
        if success:
            try:
                cursor.execute("SELECT pg_reload_conf()")
            except Exception:
                pass
        cursor.close()
        conn.close()
        return json.dumps({"success": success, "pending_restart": pending_restart, "failed": failed}, indent=2)

    def execute_simulated(self, args):
        knobs = json.loads(args["knobs"])
        for k, v in knobs.items():
            self.env_state[f"knob_{k}"] = v
        return json.dumps({"success": list(knobs.keys()), "pending_restart": [], "failed": []}, indent=2)


class RestartPGTool(DBTool):
    def __init__(self, **kwargs):
        super().__init__(
            name="restart_pg",
            description="重启 PostgreSQL 服务，使 static 参数（如 shared_buffers）生效",
            parameters={"type": "object", "properties": {}, "required": []},
            **kwargs
        )

    def execute_real(self, args):
        data_dir = self.config.get("tools.database.data_dir")
        timeout = self.config.get("tools.pg_control.restart_timeout", 30)
        try:
            start = time.time()
            cmd = f"pg_ctl -D {data_dir} restart -w -t {timeout}" if data_dir else "sudo systemctl restart postgresql"
            subprocess.run(cmd, shell=True, capture_output=True, timeout=timeout)
            return json.dumps({"success": True, "duration_seconds": round(time.time() - start, 1)})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def execute_simulated(self, args):
        return json.dumps({"success": True, "duration_seconds": 0})


class ReloadPGTool(DBTool):
    def __init__(self, **kwargs):
        super().__init__(
            name="reload_pg",
            description="重载配置（不重启），使动态参数立即生效",
            parameters={"type": "object", "properties": {}, "required": []},
            **kwargs
        )

    def execute_real(self, args):
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT pg_reload_conf()")
            cursor.close()
            conn.close()
            return json.dumps({"success": True})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def execute_simulated(self, args):
        return json.dumps({"success": True})


class ResetConfigTool(DBTool):
    def __init__(self, **kwargs):
        super().__init__(
            name="reset_config",
            description="恢复所有参数到默认值",
            parameters={"type": "object", "properties": {}, "required": []},
            **kwargs
        )

    def execute_real(self, args):
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("ALTER SYSTEM RESET ALL")
            cursor.execute("SELECT pg_reload_conf()")
            cursor.close()
            conn.close()
            return json.dumps({"success": True})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def execute_simulated(self, args):
        # 恢复到 env_state 的原始 knob（由 DBToolEnv.reset 保存）
        if hasattr(self, '_original_knobs'):
            for k, v in self._original_knobs.items():
                self.env_state[k] = v
        return json.dumps({"success": True})


# ==================== 验证类 ====================

class PredictPerformanceTool(DBTool):
    def __init__(self, cost_model=None, **kwargs):
        super().__init__(
            name="predict_performance",
            description="用 Cost Model 预测当前配置下的性能",
            parameters={"type": "object", "properties": {}, "required": []},
            **kwargs
        )
        self.cost_model = cost_model

    def execute_real(self, args):
        return json.dumps({"error": "predict_performance only available in train mode"})

    def execute_simulated(self, args):
        if self.cost_model is None:
            return json.dumps({"error": "cost model not loaded"})
        predicted = self.cost_model.predict(dict(self.env_state))
        baseline = self.env_state.get("tps", 0)
        pred_tps = predicted.get("tps", 0)
        return json.dumps({
            "predicted_tps": pred_tps,
            "predicted_latency_avg": predicted.get("latency_avg", 0),
            "baseline_tps": baseline,
            "improvement_pct": round((pred_tps - baseline) / max(baseline, 1) * 100, 2),
        }, indent=2)


class RunBenchmarkTool(DBTool):
    def __init__(self, **kwargs):
        super().__init__(
            name="run_benchmark",
            description="运行真实 benchmark（pgbench），获取实际性能",
            parameters={
                "type": "object",
                "properties": {
                    "duration": {"type": "integer", "description": "运行时长（秒），默认 60"}
                },
                "required": []
            },
            **kwargs
        )

    def execute_real(self, args):
        from cost_model.data.benchmark_runner import BenchmarkRunner
        db = self.config.database
        bench_cfg = self.config.get("tools.benchmark", {})
        runner = BenchmarkRunner(
            tool=bench_cfg.get("tool", "pgbench"),
            pg_host=db.get("host", "127.0.0.1"),
            pg_port=db.get("port", 5432),
            pg_user=db.get("user", "postgres"),
            pg_database=db.get("database", "postgres"),
            duration=args.get("duration", 60),
            clients=bench_cfg.get("clients", 8),
            threads=bench_cfg.get("threads", 4),
        )
        return json.dumps(runner.run(), indent=2)

    def execute_simulated(self, args):
        return json.dumps({"error": "run_benchmark only available in eval mode"})


# 所有工具类列表
ALL_TOOL_CLASSES = [
    GetHardwareInfoTool,
    GetCurrentConfigTool,
    GetDBMetricsTool,
    GetWorkloadInfoTool,
    GetRecentLogsTool,
    SetKnobTool,
    RestartPGTool,
    ReloadPGTool,
    ResetConfigTool,
    PredictPerformanceTool,
    RunBenchmarkTool,
]
