"""
DB 调优工具集

每个工具类包含 execute_real() 和 execute_simulated() 两个方法，
execute() 根据 mode 自动分发。
"""

import os
import json
import time
import shutil
import subprocess
import logging

from core.tool.tool_base import Tool

logger = logging.getLogger(__name__)

PENDING_RESTART_KNOBS_KEY = "__pending_restart_knobs__"


def _get_pending_restart_knobs(env_state: dict) -> dict:
    pending = env_state.get(PENDING_RESTART_KNOBS_KEY)
    if not isinstance(pending, dict):
        pending = {}
        env_state[PENDING_RESTART_KNOBS_KEY] = pending
    return pending


def _has_sudo():
    """检测是否可用 sudo（免密）"""
    if os.getuid() == 0:
        return False
    if not shutil.which('sudo'):
        return False
    try:
        r = subprocess.run(['sudo', '-n', 'true'], capture_output=True, timeout=3)
        return r.returncode == 0
    except Exception:
        return False


_SUDO_AVAILABLE = _has_sudo()


def _sudo_cmd(cmd_list):
    """自动加 sudo 前缀"""
    return (['sudo'] + cmd_list) if _SUDO_AVAILABLE else cmd_list


class DBTool(Tool):
    """DB 工具基类，扩展 real/simulated 双模式"""

    def __init__(self, name, description, parameters,
                 mode="train", config=None, env_state=None, **kwargs):
        super().__init__(name, description, parameters)
        self.mode = mode
        self.config = config
        self.env_state = env_state
        self.scenario = None  # ScenarioState，由 DBToolEnv 注入

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


class FinishTuningTool(DBTool):
    def __init__(self, **kwargs):
        super().__init__(
            name="finish_tuning",
            description="结束当前调优回合，确认保留当前配置作为最终结果",
            parameters={"type": "object", "properties": {}, "required": []},
            **kwargs
        )

    def execute_real(self, args):
        return json.dumps({"status": "finished"}, ensure_ascii=False)

    def execute_simulated(self, args):
        return json.dumps({"status": "finished"}, ensure_ascii=False)


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
        if self.scenario and self.scenario.hardware:
            info = {k: v for k, v in self.scenario.hardware.items() if v is not None and str(v) != 'nan'}
        else:
            info = {k.replace("hw_", ""): v for k, v in self.env_state.items()
                    if k.startswith("hw_") and v is not None and str(v) != 'nan'}
        return json.dumps(info, ensure_ascii=False, indent=2)

    def _cmd(self, cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True, timeout=5)
        except Exception:
            return ""


class GetCurrentConfigTool(DBTool):
    def __init__(self, tunable_knobs=None, **kwargs):
        super().__init__(
            name="get_current_config",
            description="获取当前 PostgreSQL 的 knob 配置。不传参数时返回所有可调 knob，传 knob_names 可查询指定 knob。",
            parameters={
                "type": "object",
                "properties": {
                    "knob_names": {"type": "string", "description": "要查询的 knob 名称，逗号分隔。留空返回所有可调 knob"}
                },
                "required": []
            },
            **kwargs
        )
        self.tunable_knobs = tunable_knobs or []

    def execute_real(self, args):
        conn = self._get_conn()
        cursor = conn.cursor()
        names = args.get("knob_names", "")
        if names:
            name_list = [n.strip() for n in names.split(",")]
        elif self.tunable_knobs:
            name_list = self.tunable_knobs
        else:
            # fallback：全量
            cursor.execute("SHOW ALL")
            config = {row[0]: row[1] for row in cursor.fetchall()}
            cursor.close()
            conn.close()
            return json.dumps(config, ensure_ascii=False, indent=2)

        config = {}
        for name in name_list:
            try:
                cursor.execute(f"SHOW {name}")
                config[name] = cursor.fetchone()[0]
            except Exception:
                config[name] = "unknown"
        cursor.close()
        conn.close()
        return json.dumps(config, ensure_ascii=False, indent=2)

    def execute_simulated(self, args):
        names = args.get("knob_names", "")
        if self.scenario and self.scenario.knobs:
            knobs = dict(self.scenario.knobs)
            # 未在 scenario 中列出的 knob 使用 knob_space 默认值
            if self.tunable_knobs and hasattr(self, 'knob_defaults'):
                for k in self.tunable_knobs:
                    if k not in knobs:
                        knobs[k] = self.knob_defaults.get(k, "unknown")
        else:
            knobs = {k.replace("knob_", ""): v for k, v in self.env_state.items() if k.startswith("knob_")}
        if names:
            name_list = [n.strip() for n in names.split(",")]
            knobs = {k: knobs.get(k, "unknown") for k in name_list}
        elif self.tunable_knobs:
            knobs = {k: knobs.get(k, "unknown") for k in self.tunable_knobs if k in knobs}
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
        if self.scenario and self.scenario.db_metrics:
            m = self.scenario.db_metrics
            result = {
                "buffer_hit_rate": m.get("buffer_hit_rate"),
                "temp_files_count": m.get("temp_files_count", 0),
                "temp_bytes_mb": m.get("temp_bytes_mb", 0),
                "dead_tuple_ratio": m.get("dead_tuple_ratio", 0),
                "seq_scan_ratio": m.get("seq_scan_ratio"),
                "connections": {
                    "active": m.get("active_connections", 0),
                    "idle": m.get("idle_connections", 0),
                    "idle_in_transaction": m.get("idle_in_transaction", 0),
                    "waiting": m.get("waiting_connections", 0),
                },
                "deadlocks": m.get("deadlocks", 0),
                "checkpoints_per_hour": m.get("checkpoints_per_hour"),
            }
            if self.scenario.wait_events:
                result["top_wait_events"] = self.scenario.wait_events[:5]
            # 过滤 None
            result = {k: v for k, v in result.items() if v is not None}
            return json.dumps(result, ensure_ascii=False, indent=2)

        # 兼容旧 CSV 模式
        raw = {k.replace("metric_", ""): v for k, v in self.env_state.items() if k.startswith("metric_")}
        metrics = {}
        blks_hit = float(raw.get("pg_stat_database_sum_blks_hit", 0) or 0)
        blks_read = float(raw.get("pg_stat_database_sum_blks_read", 0) or 0)
        total = blks_hit + blks_read
        if total > 0:
            metrics["buffer_hit_rate"] = round(blks_hit / total, 4)
        metrics["temp_files_count"] = int(float(raw.get("pg_stat_database_sum_temp_files", 0) or 0))
        metrics["deadlocks"] = int(float(raw.get("pg_stat_database_sum_deadlocks", 0) or 0))
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
        if self.scenario and self.scenario.workload:
            w = self.scenario.workload
            result = {
                "workload_type": w.get("type"),
                "tps_current": w.get("tps_current"),
                "latency_avg_ms": w.get("latency_avg_ms"),
                "benchmark": w.get("benchmark"),
                "clients": w.get("clients"),
            }
            if self.scenario.slow_queries:
                result["slow_queries_top3"] = self.scenario.slow_queries[:3]
            result = {k: v for k, v in result.items() if v is not None}
            return json.dumps(result, ensure_ascii=False, indent=2)

        # 兼容旧模式
        info = {k.replace("sw_", ""): v for k, v in self.env_state.items() if k.startswith("sw_")}
        return json.dumps(info, ensure_ascii=False, indent=2)


class GetRecentLogsTool(DBTool):
    def __init__(self, **kwargs):
        super().__init__(
            name="view_logs",
            description=(
                "查看 PostgreSQL 日志。\n"
                "注意：模拟模式下，这是场景初始状态的日志快照，不是实时日志。"
                "应用 knob 变更或重启后，日志不会更新。建议在诊断阶段使用，辅助判断瓶颈方向。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "返回最多 N 条日志，默认 20"},
                    "level": {"type": "string", "description": "按日志级别过滤: LOG / WARNING / ERROR / FATAL。留空返回所有级别"},
                    "keyword": {"type": "string", "description": "按关键词过滤（如 checkpoint, deadlock, temporary file）。留空不过滤"},
                },
                "required": []
            },
            **kwargs
        )

    def execute_real(self, args):
        n = args.get("n", 20)
        level = args.get("level", "")
        keyword = args.get("keyword", "")
        log_path = self.config.get("tools.database.log_path",
                                   "/var/log/postgresql/postgresql-16-main.log")
        try:
            result = subprocess.run(_sudo_cmd(["cat", log_path]),
                                    capture_output=True, text=True, timeout=5)
            lines = result.stdout.strip().split("\n")

            # 按级别过滤
            if level:
                lines = [l for l in lines if level.upper() in l]

            # 按关键词过滤
            if keyword:
                lines = [l for l in lines if keyword.lower() in l.lower()]

            return "\n".join(lines[-n:]) or "No matching log entries."
        except Exception as e:
            return f"Error reading logs: {e}"

    def execute_simulated(self, args):
        n = args.get("n", 20)
        level = args.get("level", "")
        keyword = args.get("keyword", "")

        if self.scenario and self.scenario.logs:
            logs = self.scenario.logs

            # 按级别过滤
            if level:
                level_pri = {"LOG": 0, "WARNING": 1, "ERROR": 2, "FATAL": 3}
                min_pri = level_pri.get(level.upper(), 0)
                logs = [e for e in logs
                        if level_pri.get(e.get("level", "LOG"), 0) >= min_pri]

            # 按关键词过滤
            if keyword:
                kw = keyword.lower()
                logs = [e for e in logs if kw in e.get("message", "").lower()]

            formatted = [f"{e['level']}: {e['message']}" for e in logs[-n:]]
            return "\n".join(formatted) if formatted else "No matching log entries."

        return "No log data available."


# ==================== 行动类 ====================

class SetKnobTool(DBTool):
    # 内存类 knob 名单（需要校验不超过物理内存）
    MEMORY_KNOBS = {
        "shared_buffers", "work_mem", "effective_cache_size",
        "maintenance_work_mem", "wal_buffers", "temp_buffers",
    }

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

    def _get_total_memory_gb(self) -> float:
        """获取物理内存（GB），优先从 scenario.hardware，否则读系统"""
        if self.scenario and self.scenario.hardware:
            mem = self.scenario.hardware.get("total_memory_gb")
            if mem:
                return float(mem)
        try:
            import os
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return pages * page_size / (1024 ** 3)
        except Exception:
            return 0

    def _validate_memory_knobs(self, knobs: dict) -> tuple:
        """校验内存类 knob 不超过物理内存的 80%。返回 (合法knobs, 拒绝列表)"""
        from core.db.knob_space import parse_memory
        total_mem_gb = self._get_total_memory_gb()
        if total_mem_gb <= 0:
            return knobs, []

        max_kb = int(total_mem_gb * 0.8 * 1024 * 1024)  # 80% 物理内存，单位 kB
        accepted = {}
        rejected = []
        for name, value in knobs.items():
            if name in self.MEMORY_KNOBS:
                try:
                    val_kb = parse_memory(str(value))
                    if val_kb > max_kb:
                        rejected.append({
                            "name": name,
                            "value": str(value),
                            "error": f"超过物理内存限制（{total_mem_gb:.1f}GB 的 80% = {total_mem_gb*0.8:.1f}GB）"
                        })
                        continue
                except (ValueError, TypeError):
                    pass
            accepted[name] = value
        return accepted, rejected

    def execute_real(self, args):
        knobs = json.loads(args["knobs"])

        # 校验内存类 knob
        knobs, mem_rejected = self._validate_memory_knobs(knobs)

        conn = self._get_conn()
        cursor = conn.cursor()
        success, pending_restart, failed = [], [], list(mem_rejected)
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

        # 校验内存类 knob
        knobs, mem_rejected = self._validate_memory_knobs(knobs)

        restart_knobs = set(getattr(self, "restart_knobs", set()))
        active_knobs = {}
        pending_restart = {}

        for k, v in knobs.items():
            if k in restart_knobs:
                pending_restart[k] = v
            else:
                active_knobs[k] = v

        # 动态参数立刻生效；静态参数挂到 pending_restart，等待 restart_pg 生效
        for k, v in active_knobs.items():
            self.env_state[f"knob_{k}"] = v
        if self.scenario and active_knobs:
            self.scenario.apply_knob_change(active_knobs)

        pending_state = _get_pending_restart_knobs(self.env_state)
        pending_state.update(pending_restart)

        failed = list(mem_rejected)
        return json.dumps(
            {
                "success": list(active_knobs.keys()),
                "pending_restart": list(pending_restart.keys()),
                "failed": failed,
            },
            indent=2,
        )


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
            prefix = "sudo " if _SUDO_AVAILABLE else ""
            cmd = f"pg_ctl -D {data_dir} restart -w -t {timeout}" if data_dir else f"{prefix}systemctl restart postgresql"
            subprocess.run(cmd, shell=True, capture_output=True, timeout=timeout)
            return json.dumps({"success": True, "duration_seconds": round(time.time() - start, 1)})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def execute_simulated(self, args):
        pending_knobs = dict(_get_pending_restart_knobs(self.env_state))
        if self.scenario and pending_knobs:
            self.scenario.apply_knob_change(pending_knobs)
        for k, v in pending_knobs.items():
            self.env_state[f"knob_{k}"] = v
        _get_pending_restart_knobs(self.env_state).clear()
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
        _get_pending_restart_knobs(self.env_state).clear()
        if hasattr(self, '_original_knobs'):
            for k, v in self._original_knobs.items():
                self.env_state[k] = v
            if self.scenario:
                self.scenario.knobs = {
                    key.replace("knob_", ""): value
                    for key, value in self._original_knobs.items()
                    if key.startswith("knob_")
                }
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
        self._original_knobs_snapshot = {}  # reset 时由 DBToolEnv 注入，保存未修改的原始 knob

    def execute_real(self, args):
        return json.dumps({"error": "predict_performance only available in train mode"})

    def execute_simulated(self, args):
        if self.cost_model is None:
            return json.dumps({"error": "cost model not loaded"})

        if self.scenario and self.scenario.knobs:
            hw_info = dict(self.scenario.hardware)
            wl_type = self.scenario.workload.get("type", "mixed")

            # 当前配置：scenario.knobs 已被 apply_knob_change 更新，直接用即可
            current_knobs = dict(self.scenario.knobs)
            current_knobs["workload"] = wl_type
        else:
            hw_info = {k.replace("hw_", ""): v for k, v in self.env_state.items() if k.startswith("hw_")}
            wl_type = self.env_state.get("workload", "mixed")
            current_knobs = {k.replace("knob_", ""): v for k, v in self.env_state.items() if k.startswith("knob_")}
            current_knobs["workload"] = wl_type

        # baseline 配置：使用场景原始配置（模型看到的是相对起始状态的提升）
        raw_snapshot = self._original_knobs_snapshot or {}
        baseline_knobs = {k.replace("knob_", ""): v for k, v in raw_snapshot.items()}
        baseline_knobs["workload"] = wl_type

        try:
            # 两者都用模型预测，improvement = (pred_current - pred_baseline) / pred_baseline
            pred_baseline = self.cost_model.predict(baseline_knobs, hw_info)
            pred_tps      = self.cost_model.predict(current_knobs,  hw_info)
        except Exception as e:
            return json.dumps({"error": f"prediction failed: {str(e)}"})

        IMPROVEMENT_CAP = 200.0  # 超过 200% 视为模型幻觉，截断

        raw_improvement = (pred_tps - pred_baseline) / max(pred_baseline, 1) * 100
        capped_improvement = min(IMPROVEMENT_CAP, max(0, raw_improvement))

        return json.dumps({
            "predicted_tps":     round(pred_tps, 1),
            "baseline_tps":      round(pred_baseline, 1),
            "actual_tps":        round(float(self.scenario.workload.get("tps_current", 0) or 0) if self.scenario else 0, 1),
            "improvement_pct":   round(capped_improvement, 2),
        }, indent=2)

    def calculate_reward(self, args, result):
        """从 predict_performance 返回的 improvement_pct 计算 reward"""
        try:
            parsed = json.loads(result)
            return parsed.get("improvement_pct", 0) / 100.0
        except (json.JSONDecodeError, TypeError):
            return 0.0


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
        from core.db.benchmark_runner import BenchmarkRunner
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


class GetSystemStatsTool(DBTool):
    """获取系统级资源使用情况（CPU、内存、IO、swap）"""
    def __init__(self, **kwargs):
        super().__init__(
            name="get_system_stats",
            description="获取系统级资源使用情况（CPU 使用率、内存使用率、磁盘 IO、swap 等）",
            parameters={"type": "object", "properties": {}, "required": []},
            **kwargs
        )

    def execute_real(self, args):
        stats = {}
        try:
            output = subprocess.check_output(
                "top -bn1 | grep 'Cpu(s)' | awk '{print $2}'",
                shell=True, text=True, timeout=5
            )
            stats["cpu_usage_pct"] = round(float(output.strip()), 1)
        except Exception:
            pass
        try:
            output = subprocess.check_output(
                "free -m | grep Mem | awk '{printf \"%.1f\", $3/$2*100}'",
                shell=True, text=True, timeout=5
            )
            stats["memory_usage_pct"] = float(output.strip())
        except Exception:
            pass
        try:
            output = subprocess.check_output(
                "free -m | grep Swap | awk '{print $3}'",
                shell=True, text=True, timeout=5
            )
            stats["swap_usage_mb"] = int(output.strip() or 0)
        except Exception:
            pass
        return json.dumps(stats, ensure_ascii=False, indent=2)

    def execute_simulated(self, args):
        if self.scenario and self.scenario.system:
            return json.dumps(self.scenario.system, ensure_ascii=False, indent=2)
        return json.dumps({}, ensure_ascii=False, indent=2)


# 所有工具类列表
ALL_TOOL_CLASSES = [
    GetHardwareInfoTool,
    GetCurrentConfigTool,
    GetDBMetricsTool,
    GetWorkloadInfoTool,
    GetRecentLogsTool,
    GetSystemStatsTool,
    SetKnobTool,
    RestartPGTool,
    ReloadPGTool,
    ResetConfigTool,
    PredictPerformanceTool,
    RunBenchmarkTool,
]
