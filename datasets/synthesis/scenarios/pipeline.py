"""
故障场景采集 Pipeline（两步流程）

Step 1: generate_knobs — LLM 根据种子描述生成 knob 配置（不需要 PG）
Step 2: collect_scenarios — 真机执行 + 采集完整指标（需要 PG）

用法:
    # Step 1: 生成 knob 配置
    python3 -m datasets.synthesis.scenarios.pipeline generate \
        --seeds datasets/data/scenario_seeds/seeds.json \
        --output datasets/data/scenarios/knob_configs \
        --config configs/knob_space.yaml \
        --model gpt-4 --variants 3

    # Step 2: 真机采集
    python3 -m datasets.synthesis.scenarios.pipeline collect \
        --input datasets/data/scenarios/knob_configs \
        --output datasets/data/scenarios/collected \
        --host 127.0.0.1 --port 5432
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


def create_scenario_prompt(seed: dict, hardware: dict, knob_space: KnobSpace) -> str:
    """构造让 LLM 生成故障 knob 配置的 prompt"""
    return f"""你是 PostgreSQL 故障模拟专家。请根据以下场景描述，从可用 knob 列表中选择需要修改的参数并给出值，使数据库表现出该场景的典型症状。

## 场景描述
{seed['description']}

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


# ==================== Step 1: 生成 knob 配置 ====================

def generate_knobs(seeds_path: str, output_dir: str, knob_space_path: str,
                   llm_generate, variants: int = 1, hardware: dict = None):
    """读取种子描述，调用 LLM 生成 knob 配置，保存为 JSON"""
    knob_space = KnobSpace(knob_space_path)

    with open(seeds_path, "r", encoding="utf-8") as f:
        seeds = json.load(f)

    if hardware is None:
        hardware = {"cpu_count": 8, "total_memory_gb": 16, "disk_type": "SSD"}

    os.makedirs(output_dir, exist_ok=True)
    total = len(seeds) * variants
    logger.info(f"共 {len(seeds)} 个种子 × {variants} 变体 = {total} 个配置待生成")

    generated = 0
    for seed in seeds:
        for v in range(variants):
            name = seed["name"]
            out_path = os.path.join(output_dir, f"{name}_v{v}.json")

            # 跳过已生成的
            if os.path.exists(out_path):
                logger.info(f"跳过已存在: {out_path}")
                generated += 1
                continue

            logger.info(f"[{generated+1}/{total}] 生成 {name}_v{v} ...")

            prompt = create_scenario_prompt(seed, hardware, knob_space)
            raw = llm_generate(prompt)

            # 解析
            try:
                text = raw.strip()
                if text.startswith("```"):
                    text = "\n".join(text.split("\n")[1:-1])
                knob_config = json.loads(text)
            except json.JSONDecodeError as e:
                logger.error(f"  LLM 输出非法 JSON: {e}")
                continue

            # 校验
            valid = knob_space.validate(knob_config)
            if not valid:
                logger.error(f"  所有 knob 校验失败，跳过")
                continue

            # 保存
            result = {
                "name": name,
                "variant": v,
                "difficulty": seed.get("difficulty", 1),
                "description": seed["description"],
                "category": seed.get("category", ""),
                "knobs": valid,
                "hardware_hint": hardware,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            logger.info(f"  ✅ {len(valid)} 个 knob → {out_path}")
            generated += 1

    logger.info(f"生成完成: {generated}/{total}")


# ==================== Step 2: 真机采集 ====================

def collect_scenarios(input_dir: str, output_dir: str,
                      pg_host: str = "127.0.0.1", pg_port: int = 5432,
                      pg_user: str = "postgres", pg_password: str = "",
                      pg_database: str = "postgres", pg_data_dir: str = None):
    """读取 knob 配置 JSON，逐个应用到 PG，跑 benchmark，采集完整指标"""
    pg_ctl = PGConfigurator(
        pg_host=pg_host, pg_port=pg_port,
        pg_user=pg_user, pg_password=pg_password,
        pg_database=pg_database, pg_data_dir=pg_data_dir,
    )

    knob_space_path = "configs/knob_space.yaml"
    knob_space = KnobSpace(knob_space_path)

    config_files = sorted([
        f for f in os.listdir(input_dir) if f.endswith(".json")
    ])
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"共 {len(config_files)} 个配置待采集")
    t0 = time.time()

    for i, fname in enumerate(config_files):
        out_path = os.path.join(output_dir, fname)

        # 跳过已采集的
        if os.path.exists(out_path):
            logger.info(f"跳过已采集: {fname}")
            continue

        config_path = os.path.join(input_dir, fname)
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        knobs = config["knobs"]
        logger.info(f"[{i+1}/{len(config_files)}] 采集 {config['name']}_v{config.get('variant', 0)} ...")

        try:
            # 1. 重置 → 应用
            pg_ctl.reset_to_default()
            needs_restart = knob_space.needs_restart(knobs)
            pg_ctl.apply(knobs, needs_restart=needs_restart)

            # 2. Benchmark
            benchmark = BenchmarkRunner(
                pg_host=pg_host, pg_port=pg_port,
                pg_user=pg_user, pg_database=pg_database,
            )
            perf = benchmark.run()
            logger.info(f"  TPS: {perf.get('tps', 'N/A')}")

            # 3. 采集
            import psycopg2
            conn = psycopg2.connect(
                host=pg_host, port=pg_port,
                user=pg_user, password=pg_password,
                database=pg_database
            )
            collector = ScenarioCollector(conn, database=pg_database)
            scenario_data = collector.collect_scenario()

            # 从原始指标计算人可读版本
            flat = collector.flatten_snapshot({"metrics": scenario_data.get("metrics", {}), "timestamp": ""})
            csv_metrics = {k: v for k, v in flat.items() if k.startswith("metric_")}

            from core.db.collector import HardwareCollector
            hardware = HardwareCollector().collect()
            conn.close()

            # 4. 组装 ScenarioState
            state = ScenarioState(
                name=config["name"],
                difficulty=config.get("difficulty", 1),
                description=config.get("description", ""),
                hardware={k: v for k, v in hardware.items() if v is not None},
                knobs=knobs,
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
            state.to_json(out_path)
            logger.info(f"  ✅ → {out_path}")

        except Exception as e:
            logger.error(f"  ❌ 失败: {e}")
            try:
                pg_ctl.reset_to_default()
                pg_ctl.restart()
            except Exception:
                pass

    # 恢复默认
    try:
        pg_ctl.reset_to_default()
        pg_ctl.restart()
    except Exception:
        pass

    elapsed = time.time() - t0
    logger.info(f"采集完成，耗时 {int(elapsed)}s")


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="故障场景 Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # Step 1: generate
    gen = subparsers.add_parser("generate", help="LLM 生成 knob 配置（不需要 PG）")
    gen.add_argument("--seeds", default="datasets/data/scenario_seeds/seeds.json")
    gen.add_argument("--output", default="datasets/data/scenarios/knob_configs")
    gen.add_argument("--config", default="configs/knob_space.yaml")
    gen.add_argument("--model", default="gpt-4")
    gen.add_argument("--api-key", default=None)
    gen.add_argument("--api-base", default=None)
    gen.add_argument("--variants", type=int, default=3, help="每个种子生成几个变体")

    # Step 2: collect
    col = subparsers.add_parser("collect", help="真机采集完整指标（需要 PG）")
    col.add_argument("--input", default="datasets/data/scenarios/knob_configs")
    col.add_argument("--output", default="datasets/data/scenarios/collected")
    col.add_argument("--host", default="127.0.0.1")
    col.add_argument("--port", type=int, default=5432)
    col.add_argument("--user", default="postgres")
    col.add_argument("--password", default="")
    col.add_argument("--database", default="postgres")
    col.add_argument("--pg-data-dir", default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.command == "generate":
        from datasets.synthesis.mcts.run_search import create_llm_client
        llm_fn = create_llm_client(
            model=args.model,
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
            api_base=args.api_base or os.environ.get("OPENAI_API_BASE"),
        )
        generate_knobs(
            seeds_path=args.seeds,
            output_dir=args.output,
            knob_space_path=args.config,
            llm_generate=llm_fn,
            variants=args.variants,
        )
    elif args.command == "collect":
        collect_scenarios(
            input_dir=args.input,
            output_dir=args.output,
            pg_host=args.host, pg_port=args.port,
            pg_user=args.user, pg_password=args.password,
            pg_database=args.database, pg_data_dir=args.pg_data_dir,
        )
    else:
        parser.print_help()
