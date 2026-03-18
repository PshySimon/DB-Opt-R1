"""
故障场景采集 Pipeline

流程: 读故障描述 → LLM 生成 knob → 校验 → 真机执行 → benchmark → 采集 → 存 JSON
"""

import json
import os
import logging
import time

from core.db.knob_space import KnobSpace
from core.db.pg_configurator import PGConfigurator
from core.db.benchmark_runner import BenchmarkRunner
from .scenario_collector import ScenarioCollector
from .schema import ScenarioState

logger = logging.getLogger(__name__)


def create_fault_prompt(fault_desc: dict, hardware: dict, knob_space: KnobSpace) -> str:
    """构造让 LLM 生成故障 knob 配置的 prompt"""
    return f"""你是 PostgreSQL 故障模拟专家。请根据以下故障描述，从可用 knob 列表中选择需要修改的参数并给出值，使数据库表现出该故障的典型症状。

## 故障描述
{fault_desc['description']}

## 当前硬件环境
- CPU: {hardware.get('cpu_count', 8)} 核
- 内存: {hardware.get('total_memory_gb', 16)} GB
- 磁盘: {hardware.get('disk_type', 'SSD')}

## 可用 knob 列表（只能从这些中选择，值必须在范围内）
{knob_space.summarize_for_prompt()}

## 要求
1. 只输出需要修改的 knob，其余保持 PostgreSQL 默认值
2. 值必须在上面列出的 min/max 范围内
3. 输出纯 JSON，不要任何多余文字

## 输出格式
{{"shared_buffers": "32MB", "work_mem": "1MB"}}"""


class ScenarioPipeline:
    """故障场景采集流水线"""

    def __init__(self, knob_space_path: str,
                 pg_host: str = "127.0.0.1", pg_port: int = 5432,
                 pg_user: str = "postgres", pg_password: str = "",
                 pg_database: str = "postgres", pg_data_dir: str = None,
                 output_dir: str = "./datasets/data/scenarios",
                 llm_generate=None):
        """
        Args:
            knob_space_path: knob_space.yaml 路径
            llm_generate: LLM 调用函数 (prompt) -> str
        """
        self.knob_space = KnobSpace(knob_space_path)
        self.pg_ctl = PGConfigurator(
            pg_host=pg_host, pg_port=pg_port,
            pg_user=pg_user, pg_password=pg_password,
            pg_database=pg_database, pg_data_dir=pg_data_dir,
        )
        self.output_dir = output_dir
        self.llm_generate = llm_generate

        self._pg_host = pg_host
        self._pg_port = pg_port
        self._pg_user = pg_user
        self._pg_password = pg_password
        self._pg_database = pg_database

    def _get_collector(self) -> ScenarioCollector:
        import psycopg2
        conn = psycopg2.connect(
            host=self._pg_host, port=self._pg_port,
            user=self._pg_user, password=self._pg_password,
            database=self._pg_database
        )
        return ScenarioCollector(conn, database=self._pg_database)

    def _get_benchmark(self, workload: str = "mixed") -> BenchmarkRunner:
        return BenchmarkRunner(
            pg_host=self._pg_host, pg_port=self._pg_port,
            pg_user=self._pg_user, pg_database=self._pg_database,
            workload=workload,
        )

    def run_one(self, fault_desc: dict, variant_id: int = 0) -> str:
        """对一个故障描述执行完整流程，返回 JSON 路径"""
        name = fault_desc["name"]
        logger.info(f"===== 场景: {name} (variant {variant_id}) =====")

        # 1. 采集当前硬件
        collector = self._get_collector()
        from core.db.collector import HardwareCollector
        hardware = HardwareCollector().collect()
        collector.pg_conn.close()

        # 2. LLM 生成 knob 配置
        prompt = create_fault_prompt(fault_desc, hardware, self.knob_space)
        logger.info("调用 LLM 生成故障 knob 配置...")
        raw_response = self.llm_generate(prompt)

        # 解析 JSON
        try:
            # 兼容 markdown code block
            text = raw_response.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:-1])
            knob_config = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"LLM 输出不是合法 JSON: {e}\n{raw_response}")
            return None

        # 3. 校验
        valid_knobs = self.knob_space.validate(knob_config)
        if not valid_knobs:
            logger.error("所有 knob 校验失败，跳过")
            return None
        logger.info(f"有效 knob ({len(valid_knobs)}): {list(valid_knobs.keys())}")

        # 4. 重置 PG → 应用故障配置
        self.pg_ctl.reset_to_default()
        needs_restart = self.knob_space.needs_restart(valid_knobs)
        self.pg_ctl.apply(valid_knobs, needs_restart=needs_restart)

        # 5. 跑 benchmark
        benchmark = self._get_benchmark()
        perf = benchmark.run()
        logger.info(f"TPS: {perf.get('tps', 'N/A')}")

        # 6. 采集完整场景
        collector = self._get_collector()
        scenario_data = collector.collect_scenario()
        collector.pg_conn.close()

        # 7. 组装 ScenarioState
        # 从采集数据中提取人可读的 db_metrics
        flat = collector.flatten_snapshot({"metrics": scenario_data.get("metrics", {}), "timestamp": ""})
        csv_metrics = {k: v for k, v in flat.items() if k.startswith("metric_")}

        state = ScenarioState(
            name=name,
            difficulty=fault_desc.get("difficulty", 1),
            root_cause=[],
            description=fault_desc["description"],
            hardware={k: v for k, v in hardware.items() if v is not None},
            knobs=valid_knobs,
            system=scenario_data.get("system", {}),
            db_metrics=ScenarioState._parse_csv_metrics(csv_metrics),
            wait_events=scenario_data.get("wait_events", []),
            slow_queries=scenario_data.get("slow_queries", []),
            logs=scenario_data.get("logs", []),
            workload={
                "type": "mixed",
                "tps_current": perf.get("tps", 0),
                "latency_avg_ms": perf.get("latency_avg", 0),
                "benchmark": "pgbench",
            },
            solution={},
        )

        # 8. 保存
        os.makedirs(self.output_dir, exist_ok=True)
        json_path = os.path.join(self.output_dir, f"{name}_v{variant_id}.json")
        state.to_json(json_path)

        # 9. 恢复默认
        self.pg_ctl.reset_to_default()
        self.pg_ctl.restart()

        return json_path

    def run_all(self, fault_descriptions_path: str, variants_per_fault: int = 1):
        """对所有故障描述批量运行"""
        with open(fault_descriptions_path, "r", encoding="utf-8") as f:
            faults = json.load(f)

        logger.info(f"共 {len(faults)} 个故障描述，每个 {variants_per_fault} 个变体")
        t0 = time.time()

        for fault in faults:
            for v in range(variants_per_fault):
                try:
                    path = self.run_one(fault, variant_id=v)
                    if path:
                        logger.info(f"✅ {path}")
                except Exception as e:
                    logger.error(f"❌ {fault['name']}_v{v} 失败: {e}")
                    # 尝试恢复
                    try:
                        self.pg_ctl.reset_to_default()
                        self.pg_ctl.restart()
                    except Exception:
                        pass

        elapsed = time.time() - t0
        logger.info(f"全部完成，耗时 {int(elapsed)}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="故障场景采集")
    parser.add_argument("--config", default="configs/knob_space.yaml")
    parser.add_argument("--faults", default="datasets/synthesis/scenarios/fault_descriptions/faults.json")
    parser.add_argument("--output", default="datasets/data/scenarios")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="")
    parser.add_argument("--database", default="postgres")
    parser.add_argument("--model", default="gpt-4")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--variants", type=int, default=1, help="每个故障生成几个变体")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 构造 LLM 客户端
    from datasets.synthesis.mcts.run_search import create_llm_client
    llm_fn = create_llm_client(model=args.model, api_key=args.api_key, api_base=args.api_base)

    pipeline = ScenarioPipeline(
        knob_space_path=args.config,
        pg_host=args.host, pg_port=args.port,
        pg_user=args.user, pg_password=args.password,
        pg_database=args.database,
        output_dir=args.output,
        llm_generate=llm_fn,
    )

    pipeline.run_all(args.faults, variants_per_fault=args.variants)
