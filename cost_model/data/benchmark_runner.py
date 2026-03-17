"""
负载运行器：运行 benchmark 并解析性能指标
"""

import re
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """运行 benchmark 并返回性能指标"""

    # 预定义负载类型
    WORKLOADS = {
        "mixed": {
            "desc": "TPC-B 读写混合（默认）",
            "pgbench_flags": "",
        },
        "read_only": {
            "desc": "只读查询",
            "pgbench_flags": "-S",
        },
        "high_concurrency": {
            "desc": "高并发（64连接）",
            "pgbench_flags": "",
            "clients": 64,
            "threads": 8,
        },
        "write_heavy": {
            "desc": "写密集（仅 INSERT/UPDATE）",
            "pgbench_flags": "-N",
        },
    }

    def __init__(self, tool: str = "pgbench",
                 pg_host: str = "127.0.0.1", pg_port: int = 5432,
                 pg_user: str = "postgres", pg_database: str = "postgres",
                 duration: int = 60, clients: int = 8, threads: int = 4,
                 scale_factor: int = 10, workload: str = "mixed"):
        self.tool = tool
        self.pg_host = pg_host
        self.pg_port = pg_port
        self.pg_user = pg_user
        self.pg_database = pg_database
        self.duration = duration
        self.clients = clients
        self.threads = threads
        self.scale_factor = scale_factor
        self.workload = workload

    def init_benchmark(self):
        """初始化 benchmark 数据（只需运行一次）"""
        if self.tool == "pgbench":
            self._init_pgbench()
        elif self.tool == "sysbench":
            self._init_sysbench()
        else:
            raise ValueError(f"不支持的 benchmark 工具: {self.tool}")

    def run(self) -> dict:
        """运行 benchmark 并返回性能指标

        Returns:
            {
                "tps": float,           # 每秒事务数
                "latency_avg": float,   # 平均延迟 (ms)
                "latency_p95": float,   # P95 延迟 (ms)，可能为 None
            }
        """
        if self.tool == "pgbench":
            return self._run_pgbench()
        elif self.tool == "sysbench":
            return self._run_sysbench()
        else:
            raise ValueError(f"不支持的 benchmark 工具: {self.tool}")

    # ===================== pgbench =====================

    def _init_pgbench(self):
        cmd = (
            f"pgbench -i -s {self.scale_factor} "
            f"-h {self.pg_host} -p {self.pg_port} "
            f"-U {self.pg_user} {self.pg_database}"
        )
        logger.info(f"初始化 pgbench: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"pgbench 初始化失败:\n{result.stderr}")
        logger.info("pgbench 初始化完成")

    def _run_pgbench(self) -> dict:
        wl = self.WORKLOADS.get(self.workload, self.WORKLOADS["mixed"])
        clients = wl.get("clients", self.clients)
        threads = wl.get("threads", self.threads)
        flags = wl.get("pgbench_flags", "")

        cmd = (
            f"pgbench {flags} -c {clients} -j {threads} -T {self.duration} "
            f"-h {self.pg_host} -p {self.pg_port} "
            f"-U {self.pg_user} {self.pg_database}"
        )
        logger.info(f"运行 pgbench [{self.workload}]: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                timeout=self.duration + 60)
        if result.returncode != 0:
            logger.error(f"pgbench 运行失败:\n{result.stderr}")
            return {"tps": 0, "latency_avg": 0, "latency_p95": None}

        output = result.stdout + result.stderr
        return self._parse_pgbench_output(output)

    def _parse_pgbench_output(self, output: str) -> dict:
        """解析 pgbench 输出

        示例输出:
        tps = 1234.567890 (excluding connections establishing)
        latency average = 6.477 ms
        """
        perf = {"tps": 0, "latency_avg": 0, "latency_p95": None}

        # tps
        tps_match = re.search(r"tps\s*=\s*([\d.]+)", output)
        if tps_match:
            perf["tps"] = float(tps_match.group(1))

        # latency average
        lat_match = re.search(r"latency average\s*=\s*([\d.]+)\s*ms", output)
        if lat_match:
            perf["latency_avg"] = float(lat_match.group(1))

        logger.info(f"pgbench 结果: tps={perf['tps']}, latency_avg={perf['latency_avg']}ms")
        return perf

    # ===================== sysbench =====================

    def _init_sysbench(self):
        cmd = (
            f"sysbench oltp_read_write "
            f"--db-driver=pgsql "
            f"--pgsql-host={self.pg_host} --pgsql-port={self.pg_port} "
            f"--pgsql-user={self.pg_user} --pgsql-db={self.pg_database} "
            f"--tables=10 --table-size=100000 "
            f"prepare"
        )
        logger.info(f"初始化 sysbench: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"sysbench 初始化失败:\n{result.stderr}")
        logger.info("sysbench 初始化完成")

    def _run_sysbench(self) -> dict:
        cmd = (
            f"sysbench oltp_read_write "
            f"--db-driver=pgsql "
            f"--pgsql-host={self.pg_host} --pgsql-port={self.pg_port} "
            f"--pgsql-user={self.pg_user} --pgsql-db={self.pg_database} "
            f"--tables=10 --table-size=100000 "
            f"--threads={self.threads} --time={self.duration} "
            f"run"
        )
        logger.info(f"运行 sysbench: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                timeout=self.duration + 60)
        if result.returncode != 0:
            logger.error(f"sysbench 运行失败:\n{result.stderr}")
            return {"tps": 0, "latency_avg": 0, "latency_p95": None}

        return self._parse_sysbench_output(result.stdout)

    def _parse_sysbench_output(self, output: str) -> dict:
        """解析 sysbench 输出

        示例输出:
        transactions:  12345 (205.75 per sec.)
        latency:
            avg:  4.86
            95th percentile:  9.39
        """
        perf = {"tps": 0, "latency_avg": 0, "latency_p95": None}

        tps_match = re.search(r"transactions:\s+\d+\s+\(([\d.]+)\s+per sec", output)
        if tps_match:
            perf["tps"] = float(tps_match.group(1))

        lat_avg_match = re.search(r"avg:\s+([\d.]+)", output)
        if lat_avg_match:
            perf["latency_avg"] = float(lat_avg_match.group(1))

        lat_p95_match = re.search(r"95th percentile:\s+([\d.]+)", output)
        if lat_p95_match:
            perf["latency_p95"] = float(lat_p95_match.group(1))

        logger.info(f"sysbench 结果: tps={perf['tps']}, latency_avg={perf['latency_avg']}ms")
        return perf
