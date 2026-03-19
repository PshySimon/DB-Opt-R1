"""
故障场景采集 Pipeline（三步流程）

Step 0: seeds     — LLM 生成种子场景描述（可选，用于扩充种子）
Step 1: generate  — LLM 根据种子生成 knob 配置（不需要 PG）
Step 2: collect   — 真机执行 + 采集完整指标（需要 PG）

用法:
    # Step 0: LLM 扩充种子（可选）
    python3 -m datasets.synthesis.scenarios.pipeline seeds \
        --config configs/knob_space.yaml \
        --output datasets/data/scenario_seeds/seeds.json \
        --count 50 --model gpt-4

    # Step 1: 生成 knob 配置
    python3 -m datasets.synthesis.scenarios.pipeline generate \
        --seeds datasets/data/scenario_seeds/seeds.json \
        --output datasets/data/scenarios/knob_configs.json \
        --config configs/knob_space.yaml \
        --model gpt-4 --variants 3

    # Step 2: 真机采集
    python3 -m datasets.synthesis.scenarios.pipeline collect \
        --input datasets/data/scenarios/knob_configs.json \
        --output datasets/data/scenarios/collected.json \
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


SEED_GENERATION_PROMPT = """你是 PostgreSQL 性能调优专家。请生成 {count} 个不同的数据库性能问题场景描述，用于训练 AI 调优助手。

## 可用的 knob（参数）列表
{knob_summary}

## 已有种子（避免重复）
{existing}

## 要求
1. 每个场景必须能通过修改上面列出的 knob 来触发或缓解
2. 覆盖不同类别：内存、WAL、优化器、vacuum、并行、连接、锁、bgwriter、检查点、组合故障等
3. 难度 1（单参数调整）到 3（多参数组合）均匀分布
4. 描述要具体，包含故障的典型现象（如"大量临时文件"、"checkpoint 过于频繁"）
5. 输出纯 JSON 数组，不要任何多余文字

## 输出格式
[
  {{
    "name": "场景英文短名（snake_case）",
    "difficulty": 1-3,
    "description": "具体的中文场景描述，包含故障现象和根因",
    "category": "类别英文短名"
  }},
  ...
]"""


# ==================== Step 0: LLM 生成种子 ====================

def generate_seeds(knob_space_path: str, output_path: str, llm_generate,
                   count: int = 50, existing_path: str = None):
    """让 LLM 生成种子场景描述"""
    knob_space = KnobSpace(knob_space_path)

    # 加载已有种子
    existing = []
    if existing_path and os.path.exists(existing_path):
        with open(existing_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    elif output_path and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    existing_summary = "\n".join(
        f"- {s['name']}: {s['description'][:50]}..." for s in existing
    ) if existing else "（暂无）"

    # 分批生成（每批 20 个，避免输出太长截断）
    batch_size = min(count, 20)
    all_new = []

    while len(all_new) < count:
        remaining = count - len(all_new)
        n = min(batch_size, remaining)

        prompt = SEED_GENERATION_PROMPT.format(
            count=n,
            knob_summary=knob_space.summarize_for_prompt(),
            existing=existing_summary,
        )

        logger.info(f"请求 LLM 生成 {n} 个种子...")
        raw = llm_generate(prompt)

        try:
            text = raw.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:-1])
            batch = json.loads(text)
            if not isinstance(batch, list):
                logger.error("LLM 输出不是数组")
                continue
        except json.JSONDecodeError as e:
            logger.error(f"LLM 输出非法 JSON: {e}")
            continue

        # 去重
        existing_names = {s["name"] for s in existing + all_new}
        for s in batch:
            if s.get("name") and s["name"] not in existing_names:
                all_new.append(s)
                existing_names.add(s["name"])
                existing_summary += f"\n- {s['name']}: {s.get('description', '')[:50]}..."

        logger.info(f"  本批有效 {len(batch)} 个，累计 {len(all_new)} 个")

    # 合并保存
    merged = existing + all_new[:count]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 共 {len(merged)} 个种子 → {output_path}")


# ==================== Step 1: 生成 knob 配置 ====================

WORKLOAD_TYPES = ["mixed", "read_only", "write_heavy", "high_concurrency"]

def generate_knobs(seeds_path: str, output_path: str, knob_space_path: str,
                   llm_generate, variants: int = 1, hardware: dict = None,
                   workers: int = 1):
    """读取种子描述，调用 LLM 生成 knob 配置，保存为单个 JSON 文件"""
    import threading
    import random as _random

    knob_space = KnobSpace(knob_space_path)

    with open(seeds_path, "r", encoding="utf-8") as f:
        seeds = json.load(f)

    if hardware is None:
        hardware = {"cpu_count": 8, "total_memory_gb": 16, "disk_type": "SSD"}

    # 加载已有结果（断点续跑）
    results = []
    existing_keys = set()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        existing_keys = {(e["name"], e["variant"]) for e in results}
        logger.info(f"已有 {len(results)} 条记录，断点续跑")

    # 构建待生成任务
    tasks = []
    for seed in seeds:
        for v in range(variants):
            if (seed["name"], v) not in existing_keys:
                tasks.append((seed, v))

    total_all = len(seeds) * variants
    logger.info(f"共 {total_all} 个（已有 {len(results)}），待生成 {len(tasks)} 个，并发 {workers}")

    if not tasks:
        logger.info("无需生成")
        return

    # 线程安全的进度计数器 + 增量保存
    lock = threading.Lock()
    counter = {"done": 0, "success": 0, "total": len(tasks)}

    def _save():
        """写磁盘（调用方需持有 lock）"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def _generate_one(args):
        seed, v = args
        name = seed["name"]
        try:
            prompt = create_scenario_prompt(seed, hardware, knob_space)
            raw = llm_generate(prompt)

            text = raw.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:-1])
            knob_config = json.loads(text)

            valid = knob_space.validate(knob_config)
            if not valid:
                with lock:
                    counter["done"] += 1
                    logger.error(f"  [{counter['done']}/{counter['total']}] ❌ {name}_v{v}: 校验失败")
                return

            # 为场景分配负载类型
            workload = _random.choice(WORKLOAD_TYPES)

            result = {
                "name": name,
                "variant": v,
                "difficulty": seed.get("difficulty", 1),
                "description": seed["description"],
                "category": seed.get("category", ""),
                "workload": workload,
                "knobs": valid,
                "hardware_hint": hardware,
            }

            with lock:
                results.append(result)
                counter["done"] += 1
                counter["success"] += 1
                logger.info(f"  [{counter['done']}/{counter['total']}] ✅ {name}_v{v}: {len(valid)} knob, workload={workload}")
                # 每条都写磁盘
                _save()

        except json.JSONDecodeError as e:
            with lock:
                counter["done"] += 1
                logger.error(f"  [{counter['done']}/{counter['total']}] ❌ {name}_v{v}: 非法 JSON: {e}")
        except Exception as e:
            with lock:
                counter["done"] += 1
                logger.error(f"  [{counter['done']}/{counter['total']}] ❌ {name}_v{v}: {e}")

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_generate_one, t) for t in tasks]
            for future in as_completed(futures):
                future.result()  # 触发异常
    else:
        for t in tasks:
            _generate_one(t)

    logger.info(f"生成完成: 成功 {counter['success']}/{counter['total']}，总计 {len(results)} 条 → {output_path}")


# ==================== Step 2: 真机采集 ====================

def collect_scenarios(input_path: str, output_path: str,
                      pg_host: str = "127.0.0.1", pg_port: int = 5432,
                      pg_user: str = "postgres", pg_password: str = "",
                      pg_database: str = "postgres", pg_data_dir: str = None,
                      knob_space_path: str = "configs/knob_space.yaml"):
    """读取 knob 配置 JSON，逐个应用到 PG，跑 benchmark，采集完整指标"""
    pg_ctl = PGConfigurator(
        pg_host=pg_host, pg_port=pg_port,
        pg_user=pg_user, pg_password=pg_password,
        pg_database=pg_database, pg_data_dir=pg_data_dir,
    )
    knob_space = KnobSpace(knob_space_path)

    with open(input_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    # 加载已有结果（断点续跑）
    existing = []
    existing_keys = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing_keys = {(e["name"], e.get("variant", 0)) for e in existing
                        if isinstance(e, dict)}

    pending = [c for c in configs if (c["name"], c.get("variant", 0)) not in existing_keys]
    logger.info(f"共 {len(configs)} 个配置，已采集 {len(existing)}，待采集 {len(pending)}")

    results = list(existing)
    t0 = time.time()

    for i, config in enumerate(pending):
        knobs = config["knobs"]
        workload = config.get("workload", "mixed")
        label = f"{config['name']}_v{config.get('variant', 0)}"
        logger.info(f"[{i+1}/{len(pending)}] 采集 {label} (workload={workload}) ...")

        try:
            pg_ctl.reset_to_default()
            needs_restart = knob_space.needs_restart(knobs)
            pg_ctl.apply(knobs, needs_restart=needs_restart)

            benchmark = BenchmarkRunner(
                pg_host=pg_host, pg_port=pg_port,
                pg_user=pg_user, pg_database=pg_database,
                workload=workload,
            )
            perf = benchmark.run()
            logger.info(f"  TPS: {perf.get('tps', 'N/A')}")

            import psycopg2
            conn = psycopg2.connect(
                host=pg_host, port=pg_port,
                user=pg_user, password=pg_password,
                database=pg_database
            )
            collector = ScenarioCollector(conn, database=pg_database)
            scenario_data = collector.collect_scenario()

            flat = collector.flatten_snapshot({"metrics": scenario_data.get("metrics", {}), "timestamp": ""})
            csv_metrics = {k: v for k, v in flat.items() if k.startswith("metric_")}

            from core.db.collector import HardwareCollector
            hardware = HardwareCollector().collect()
            conn.close()

            from dataclasses import asdict
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
            results.append(asdict(state))
            logger.info(f"  ✅ {label}")

            # 每条保存一次（防丢失）
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"  ❌ {label}: {e}")
            try:
                pg_ctl.reset_to_default()
                pg_ctl.restart()
            except Exception:
                pass

    try:
        pg_ctl.reset_to_default()
        pg_ctl.restart()
    except Exception:
        pass

    elapsed = time.time() - t0
    logger.info(f"采集完成: {len(results)} 条，耗时 {int(elapsed)}s → {output_path}")


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="故障场景 Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # Step 0: seeds
    sd = subparsers.add_parser("seeds", help="LLM 生成种子场景描述")
    sd.add_argument("--config", default="configs/knob_space.yaml")
    sd.add_argument("--output", default="datasets/data/scenario_seeds/seeds.json")
    sd.add_argument("--existing", default=None, help="已有种子文件（用于去重）")
    sd.add_argument("--count", type=int, default=50, help="生成数量")
    sd.add_argument("--model", default="gpt-5")
    sd.add_argument("--api-key", default=None)
    sd.add_argument("--api-base", default=None)

    # Step 1: generate
    gen = subparsers.add_parser("generate", help="LLM 生成 knob 配置（不需要 PG）")
    gen.add_argument("--seeds", default="datasets/data/scenario_seeds/seeds.json")
    gen.add_argument("--output", default="datasets/data/scenarios/knob_configs.json")
    gen.add_argument("--config", default="configs/knob_space.yaml")
    gen.add_argument("--model", default="gpt-5")
    gen.add_argument("--api-key", default=None)
    gen.add_argument("--api-base", default=None)
    gen.add_argument("--variants", type=int, default=3, help="每个种子生成几个变体")
    gen.add_argument("--workers", type=int, default=5, help="并发线程数")
    gen.add_argument("--cpu", type=int, default=8, help="CPU 核数")
    gen.add_argument("--memory", type=int, default=16, help="内存 GB")
    gen.add_argument("--disk", default="SSD", choices=["SSD", "HDD", "NVMe"], help="磁盘类型")

    # Step 2: collect
    col = subparsers.add_parser("collect", help="真机采集完整指标（需要 PG）")
    col.add_argument("--input", default="datasets/data/scenarios/knob_configs.json")
    col.add_argument("--output", default="datasets/data/scenarios/collected.json")
    col.add_argument("--config", default="configs/knob_space.yaml")
    col.add_argument("--host", default="127.0.0.1")
    col.add_argument("--port", type=int, default=5432)
    col.add_argument("--user", default="postgres")
    col.add_argument("--password", default="")
    col.add_argument("--database", default="postgres")
    col.add_argument("--pg-data-dir", default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.command == "seeds":
        from datasets.synthesis.mcts.run_search import create_llm_client
        llm_fn = create_llm_client(
            model=args.model,
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
            api_base=args.api_base or os.environ.get("OPENAI_API_BASE"),
        )
        generate_seeds(
            knob_space_path=args.config,
            output_path=args.output,
            llm_generate=llm_fn,
            count=args.count,
            existing_path=args.existing,
        )

    elif args.command == "generate":
        from datasets.synthesis.mcts.run_search import create_llm_client
        llm_fn = create_llm_client(
            model=args.model,
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
            api_base=args.api_base or os.environ.get("OPENAI_API_BASE"),
        )
        generate_knobs(
            seeds_path=args.seeds,
            output_path=args.output,
            knob_space_path=args.config,
            llm_generate=llm_fn,
            variants=args.variants,
            workers=args.workers,
            hardware={"cpu_count": args.cpu, "total_memory_gb": args.memory, "disk_type": args.disk},
        )

    elif args.command == "collect":
        collect_scenarios(
            input_path=args.input,
            output_path=args.output,
            pg_host=args.host, pg_port=args.port,
            pg_user=args.user, pg_password=args.password,
            pg_database=args.database, pg_data_dir=args.pg_data_dir,
            knob_space_path=args.config,
        )

    else:
        parser.print_help()
