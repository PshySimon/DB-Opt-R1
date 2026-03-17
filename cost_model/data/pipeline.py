"""
数据采集 Pipeline：串联各模块，自动化循环采集
"""

import logging
import signal
import subprocess
import time
from pathlib import Path

from .collector import DataCollector
from .knob_generator import KnobSpace, KnobGenerator
from .pg_configurator import PGConfigurator
from .benchmark_runner import BenchmarkRunner

logger = logging.getLogger(__name__)


def _cleanup_on_exit(pg_ctl: PGConfigurator):
    """进程退出时清理：杀 pgbench、恢复 PG 配置"""
    logger.info("收到终止信号，正在清理...")

    # 1. 杀掉所有 pgbench 进程
    try:
        subprocess.run(["pkill", "-f", "pgbench"], capture_output=True, timeout=5)
        logger.info("  已停止 pgbench")
    except Exception:
        pass

    # 2. 重置 PG 配置
    try:
        pg_ctl.reset_to_default()
        pg_ctl.restart()
        logger.info("  已恢复 PG 默认配置并重启")
    except Exception:
        try:
            pg_ctl.force_reset()
            pg_ctl.restart()
            logger.info("  已强制恢复 PG 并重启")
        except Exception as e:
            logger.error(f"  PG 恢复失败: {e}")

    logger.info("清理完成，退出")


class Pipeline:
    """数据采集 Pipeline"""

    def __init__(self, config_path: str,
                 pg_host: str = "127.0.0.1", pg_port: int = 5432,
                 pg_user: str = "postgres", pg_password: str = "",
                 pg_database: str = "postgres", pg_data_dir: str = None,
                 output_dir: str = "./cost_model/data/raw",
                 seed: int = None, workload: str = "mixed"):

        # 加载配置
        self.knob_space = KnobSpace(config_path)
        bench_cfg = self.knob_space.benchmark
        coll_cfg = self.knob_space.collection

        # 初始化各模块
        self.knob_gen = KnobGenerator(self.knob_space, seed=seed)

        self.pg_ctl = PGConfigurator(
            pg_host=pg_host, pg_port=pg_port,
            pg_user=pg_user, pg_password=pg_password,
            pg_database=pg_database, pg_data_dir=pg_data_dir,
        )

        self.benchmark = BenchmarkRunner(
            tool=bench_cfg.get("tool", "pgbench"),
            pg_host=pg_host, pg_port=pg_port,
            pg_user=pg_user, pg_database=pg_database,
            duration=bench_cfg.get("duration", 60),
            clients=bench_cfg.get("clients", 8),
            threads=bench_cfg.get("threads", 4),
            scale_factor=bench_cfg.get("scale_factor", 10),
            workload=workload,
        )
        self.workload = workload

        self.output_dir = output_dir
        self.num_rounds = coll_cfg.get("num_rounds", 100)

        # collector 延迟初始化（需要 DB 连接）
        self._pg_host = pg_host
        self._pg_port = pg_port
        self._pg_user = pg_user
        self._pg_password = pg_password
        self._pg_database = pg_database

    def _get_collector(self) -> DataCollector:
        import psycopg2
        conn = psycopg2.connect(
            host=self._pg_host, port=self._pg_port,
            user=self._pg_user, password=self._pg_password,
            database=self._pg_database
        )
        return DataCollector(conn, database=self._pg_database)

    def init(self):
        """初始化 benchmark 数据（只需运行一次）"""
        logger.info("初始化 benchmark 数据...")
        self.benchmark.init_benchmark()
        logger.info("初始化完成")

    def run(self, num_rounds: int = None, sampling: str = "random"):
        """运行数据采集

        Args:
            num_rounds: 采集轮数，默认使用配置文件中的值
            sampling: 采样策略，"random" 或 "lhs"
        """
        # 注册退出清理信号
        def _signal_handler(sig, frame):
            _cleanup_on_exit(self.pg_ctl)
            exit(0)

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
        n = num_rounds or self.num_rounds

        # 生成所有 knob 配置
        if sampling == "lhs":
            configs = self.knob_gen.sample_lhs(n)
        else:
            configs = [self.knob_gen.sample_random() for _ in range(n)]

        logger.info(f"开始采集 {n} 轮数据，采样策略: {sampling}")
        logger.info(f"预计耗时: ~{n * 70 // 60} 分钟")

        success_count = 0
        fail_count = 0

        for i, knob_config in enumerate(configs):
            round_start = time.time()
            logger.info(f"===== 第 {i+1}/{n} 轮 =====")

            try:
                row = self._run_one_round(knob_config)
                if row:
                    row["status"] = "success"
                    collector = self._get_collector()
                    collector.save_csv(row, self.output_dir)
                    collector.pg_conn.close()
                    success_count += 1

                elapsed = time.time() - round_start
                logger.info(f"第 {i+1} 轮完成，耗时 {elapsed:.1f}s，"
                           f"tps={row.get('tps', 'N/A')}")

            except Exception as e:
                fail_count += 1
                error_msg = str(e)
                logger.error(f"第 {i+1} 轮失败: {error_msg}")

                # 判断失败类型
                if "restart" in error_msg.lower() or "start" in error_msg.lower():
                    status = "restart_failed"
                else:
                    status = "benchmark_failed"

                # 保存失败数据（只有 knob 配置 + 状态，tps=0）
                try:
                    fail_row = self._build_fail_row(knob_config, status, error_msg)
                    collector = self._get_collector()
                    collector.save_csv(fail_row, self.output_dir)
                    collector.pg_conn.close()
                    logger.info(f"  失败数据已保存 (status={status})")
                except Exception:
                    logger.warning("  保存失败数据也失败，跳过")

                # 尝试恢复默认配置
                try:
                    self.pg_ctl.reset_to_default()
                    self.pg_ctl.restart()
                    logger.info("  ✓ 已恢复默认配置")
                except Exception:
                    # PG 挂了，强制清文件重启
                    try:
                        logger.warning("  SQL 恢复失败，强制清配置文件重启...")
                        self.pg_ctl.force_reset()
                        self.pg_ctl.restart()
                        logger.info("  ✓ 强制恢复成功")
                    except Exception as re:
                        logger.error(f"  强制恢复也失败: {re}，跳过本轮继续")
                        continue

        logger.info(f"采集完成: 成功 {success_count} 轮，失败 {fail_count} 轮")
        logger.info(f"数据保存在: {self.output_dir}/dataset.csv")

    def _run_one_round(self, knob_config: dict) -> dict:
        """执行一轮采集"""

        # 1. 判断是否需要重启
        needs_restart = self.knob_space.needs_restart(knob_config)

        # 2. 应用配置
        self.pg_ctl.apply(knob_config, needs_restart=needs_restart)

        # 3. 采集 before
        collector = self._get_collector()
        snapshot_before = collector.collect_snapshot()

        # 4. 跑负载
        perf = self.benchmark.run()

        # 5. 采集 after
        snapshot_after = collector.collect_snapshot()
        collector.pg_conn.close()

        # 6. 计算 diff，组装一行
        diff_data = collector.collect_diff(snapshot_before, snapshot_after)
        flat = collector.flatten_snapshot(diff_data)

        # 追加性能指标和负载类型作为 label
        flat["status"] = "success"
        flat["workload"] = self.workload
        flat["tps"] = perf["tps"]
        flat["latency_avg"] = perf["latency_avg"]
        flat["latency_p95"] = perf.get("latency_p95")

        return flat

    def _build_fail_row(self, knob_config: dict, status: str, error_msg: str) -> dict:
        """构造失败数据行：保留 knob 配置 + 硬件信息 + 失败标记"""
        from .collector import HardwareCollector

        flat = {}
        flat["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        # 硬件特征（不依赖 PG 连接）
        hw = HardwareCollector().collect()
        for k, v in hw.items():
            flat[f"hw_{k}"] = v

        # 记录本轮设置的 knob（只记我们调的参数）
        for k, v in knob_config.items():
            flat[f"knob_{k}"] = v

        # 标记
        flat["status"] = status
        flat["error"] = error_msg[:200]  # 截断过长的错误信息
        flat["workload"] = self.workload
        flat["tps"] = 0
        flat["latency_avg"] = 0
        flat["latency_p95"] = None

        return flat


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据采集 Pipeline")
    parser.add_argument("--config", default="configs/knob_space.yaml",
                        help="Knob 搜索空间配置")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="")
    parser.add_argument("--database", default="postgres")
    parser.add_argument("--pg-data-dir", default=None,
                        help="PG 数据目录（用于 pg_ctl restart）")
    parser.add_argument("--output", default="./cost_model/data/raw")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--sampling", choices=["random", "lhs"], default="random")
    parser.add_argument("--init", action="store_true",
                        help="初始化 benchmark 数据")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workload", default="mixed",
                        choices=["mixed", "read_only", "high_concurrency", "write_heavy", "all"],
                        help="负载类型，all=每轮随机选一种")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    if args.workload == "all":
        # 每种负载各跑 rounds/4 轮
        import random
        workloads = ["mixed", "read_only", "high_concurrency", "write_heavy"]
        rounds_per = max(1, (args.rounds or 100) // len(workloads))
        for wl in workloads:
            logger.info(f"\n{'#'*50}")
            logger.info(f"  负载类型: {wl} ({rounds_per} 轮)")
            logger.info(f"{'#'*50}")
            p = Pipeline(
                config_path=args.config,
                pg_host=args.host, pg_port=args.port,
                pg_user=args.user, pg_password=args.password,
                pg_database=args.database, pg_data_dir=args.pg_data_dir,
                output_dir=args.output, seed=args.seed, workload=wl,
            )
            if args.init:
                p.init()
                args.init = False  # 只初始化一次
            p.run(num_rounds=rounds_per, sampling=args.sampling)
    else:
        pipeline = Pipeline(
            config_path=args.config,
            pg_host=args.host, pg_port=args.port,
            pg_user=args.user, pg_password=args.password,
            pg_database=args.database, pg_data_dir=args.pg_data_dir,
            output_dir=args.output, seed=args.seed, workload=args.workload,
        )

        if args.init:
            pipeline.init()

        pipeline.run(num_rounds=args.rounds, sampling=args.sampling)
