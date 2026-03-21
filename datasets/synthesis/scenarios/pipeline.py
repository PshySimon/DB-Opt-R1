"""
场景采集 Pipeline（三步流程）

Step 0: seeds     — 按瓶颈方向让 LLM 生成种子场景描述（或程序化生成）
Step 1: generate  — LLM 根据种子生成可用但有优化空间的 knob 配置
Step 2: collect   — 真机执行 + 采集完整指标（需要 PG）

用法:
    # Step 0: 按瓶颈方向生成种子
    python3 -m datasets.synthesis.scenarios.pipeline seeds \
        --effects configs/knob_effects.yaml \
        --knob-space configs/knob_space.yaml \
        --output datasets/data/scenarios/seeds.json \
        --count 100 --model gpt-5

    # Step 1: LLM 生成 knob 配置
    python3 -m datasets.synthesis.scenarios.pipeline generate \
        --seeds datasets/data/scenarios/seeds.json \
        --output datasets/data/scenarios/knob_configs.json \
        --config configs/knob_space.yaml \
        --model gpt-5 --variants 5 --workers 5 \
        --cpu 8 --memory 16 --disk HDD

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
    """构造让 LLM 生成调优场景 knob 配置的 prompt"""
    return f"""你是 PostgreSQL 性能调优专家。请根据以下瓶颈场景描述，给出一组合理的 knob 配置。

## 瓶颈场景
{seed['description']}

## 硬件环境
- CPU: {hardware.get('cpu_count', 8)} 核
- 内存: {hardware.get('total_memory_gb', 16)} GB
- 磁盘: {hardware.get('disk_type', 'SSD')}

## 可用 knob 列表（只能从这些中选择，值必须在范围内）
{knob_space.summarize_for_prompt()}

## 要求
1. 给出的配置必须是**可用的**（数据库能正常启动和运行）
2. 配置应体现该瓶颈场景的特征：性能不差到不可用，但**存在明显优化空间**
3. 只输出需要修改的 knob（2-5 个），其余保持默认
4. 值要合理，不要极端值（不要直接用 min 或 max）
5. 输出纯 JSON，不要任何多余文字

## 输出格式
{{"shared_buffers": "256MB", "work_mem": "2MB"}}"""


# 按瓶颈方向生成种子的 prompt
SEED_BY_DIRECTION_PROMPT = """你是 PostgreSQL 性能调优专家。请针对【{direction}】瓶颈方向，生成 {count} 个不同的性能瓶颈场景。

## 瓶颈方向说明
{direction_desc}

## 该方向涉及的 knob
{knob_list}

## 已有种子（避免重复）
{existing}

## 要求
1. 每个场景描述一个**可用但性能不佳**的配置状况，而非极端故障
2. 描述要具体，包含瓶颈的典型现象
3. 场景应有调优空间：通过调整上述 knob 可以显著改善性能
4. 难度 1（单参数调整）到 3（多参数组合）均匀分布
5. 输出纯 JSON 数组

## 输出格式
[
  {{
    "name": "场景英文短名（snake_case）",
    "difficulty": 1-3,
    "description": "具体的中文场景描述，包含瓶颈现象和可优化方向",
    "category": "{category}"
  }},
  ...
]"""


# ==================== Step 0: 按瓶颈方向生成种子 ====================

# 8 个瓶颈方向及其描述
BOTTLENECK_DIRECTIONS = {
    "memory": "内存配置瓶颈：shared_buffers/work_mem/effective_cache_size 等配置不当，缓存命中率低、排序溢写、内存分配不合理",
    "optimizer": "优化器代价估算瓶颈：random_page_cost/statistics_target/join_collapse_limit 等配置不当，执行计划选择次优",
    "wal": "WAL/Checkpoint 瓶颈：max_wal_size/checkpoint_timeout/synchronous_commit 等配置不当，写入性能受限或恢复风险高",
    "vacuum": "VACUUM/AutoVacuum 瓶颈：autovacuum 参数配置不当，表膨胀、统计信息过时、死元组堆积",
    "parallel": "并行查询瓶颈：max_parallel_workers 等配置不当，无法充分利用多核 CPU",
    "bgwriter": "后台写入瓶颈：bgwriter_delay/lru_maxpages 等配置不当，脏页清理效率低或 IO 争抢",
    "connections": "连接与资源瓶颈：max_connections/idle_in_transaction_session_timeout 等配置不当，连接资源浪费或不足",
    "locks": "锁与并发瓶颈：deadlock_timeout/lock_timeout 配置不当，锁等待策略影响并发性能",
}


def _extract_knob_list_for_direction(effects_data: dict, direction: str) -> str:
    """从 knob_effects.yaml 中提取某个方向涉及的 knob 及其效果描述"""
    knobs_info = effects_data.get("knobs", {})
    lines = []
    for knob_name, info in knobs_info.items():
        if info.get("category") == direction:
            parts = [f"  {knob_name}:"]
            if info.get("too_low"):
                parts.append(f"偏低时 → {info['too_low']}")
            if info.get("too_high"):
                parts.append(f"偏高时 → {info['too_high']}")
            if info.get("off"):
                parts.append(f"关闭时 → {info['off']}")
            lines.append("；".join(parts))
    return "\n".join(lines) if lines else "（该方向暂无详细 knob 信息）"


def generate_seeds(effects_path: str, knob_space_path: str, output_path: str,
                   llm_generate, count: int = 100):
    """按瓶颈方向让 LLM 生成种子场景描述

    Args:
        effects_path: knob_effects.yaml 路径
        knob_space_path: knob_space.yaml 路径
    """
    import yaml
    with open(effects_path, "r", encoding="utf-8") as f:
        effects_data = yaml.safe_load(f)

    # 加载已有种子
    existing = []
    if output_path and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    # 每个方向分配的数量
    directions = list(BOTTLENECK_DIRECTIONS.keys())
    per_direction = max(1, count // len(directions))

    all_new = []
    existing_names = {s["name"] for s in existing}

    for direction in directions:
        direction_desc = BOTTLENECK_DIRECTIONS[direction]
        knob_list = _extract_knob_list_for_direction(effects_data, direction)

        existing_summary = "\n".join(
            f"- {s['name']}: {s.get('description', '')[:50]}..."
            for s in existing + all_new
        ) if (existing or all_new) else "（暂无）"

        prompt = SEED_BY_DIRECTION_PROMPT.format(
            direction=direction,
            direction_desc=direction_desc,
            knob_list=knob_list,
            count=per_direction,
            existing=existing_summary,
            category=direction,
        )

        logger.info(f"[{direction}] 请求 LLM 生成 {per_direction} 个种子...")
        try:
            raw = llm_generate(prompt)
            text = raw.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:-1])
            batch = json.loads(text)
            if not isinstance(batch, list):
                logger.error(f"[{direction}] LLM 输出不是数组")
                continue
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"[{direction}] LLM 生成失败: {e}")
            continue

        # 去重
        added = 0
        for s in batch:
            if s.get("name") and s["name"] not in existing_names:
                s.setdefault("category", direction)
                all_new.append(s)
                existing_names.add(s["name"])
                added += 1

        logger.info(f"[{direction}] 新增 {added} 个，累计 {len(all_new)} 个")

    # 保存
    merged = existing + all_new
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
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
    """读取 knob 配置 JSON，逐个应用到 PG，跑 benchmark，采集完整指标

    input_path 支持 glob 模式（如 knob_configs_*.json），会合并所有匹配文件。
    """
    import glob as glob_mod

    pg_ctl = PGConfigurator(
        pg_host=pg_host, pg_port=pg_port,
        pg_user=pg_user, pg_password=pg_password,
        pg_database=pg_database, pg_data_dir=pg_data_dir,
    )
    knob_space = KnobSpace(knob_space_path)

    # 支持 glob 合并多个文件
    matched_files = sorted(glob_mod.glob(input_path))
    if not matched_files:
        # 不是 glob 模式，当作单文件
        matched_files = [input_path]

    configs = []
    for fpath in matched_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"  加载 {fpath}: {len(data)} 条")
        configs.extend(data)
    logger.info(f"合计加载 {len(configs)} 条配置")

    # 按 knobs 内容去重
    seen_knobs = set()
    unique_configs = []
    dup_count = 0
    for c in configs:
        knob_key = json.dumps(c["knobs"], sort_keys=True)
        if knob_key not in seen_knobs:
            seen_knobs.add(knob_key)
            unique_configs.append(c)
        else:
            dup_count += 1
    if dup_count > 0:
        logger.info(f"去重: {len(configs)} → {len(unique_configs)} 条（移除 {dup_count} 条重复配置，{dup_count*100/len(configs):.1f}%）")
    configs = unique_configs

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

            # scenario_data 已包含 hardware/software/knobs/metrics 等完整快照
            flat = collector.flatten_snapshot(scenario_data)
            csv_metrics = {k: v for k, v in flat.items() if k.startswith("metric_")}

            hardware = scenario_data.get("hardware", {})

            # 从 PG 查全量 tunable knob 的当前值
            cursor = conn.cursor()
            all_knobs = {}
            for knob_name in knob_space.knobs:
                try:
                    cursor.execute(f"SHOW {knob_name}")
                    all_knobs[knob_name] = cursor.fetchone()[0]
                except Exception:
                    all_knobs[knob_name] = knobs.get(knob_name, "unknown")
            cursor.close()
            conn.close()

            from dataclasses import asdict
            state = ScenarioState(
                name=config["name"],
                variant=config.get("variant", 0),
                source=config.get("source", "llm_generated"),
                difficulty=config.get("difficulty", 1),
                description=config.get("description", ""),
                hardware={k: v for k, v in hardware.items() if v is not None},
                knobs=all_knobs,
                system=scenario_data.get("system", {}),
                db_metrics=ScenarioState._parse_csv_metrics(csv_metrics),
                wait_events=scenario_data.get("wait_events", []),
                slow_queries=scenario_data.get("slow_queries", []),
                logs=scenario_data.get("logs", []),
                workload={
                    "type": workload,
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
                pg_ctl.safe_restart()  # force_reset → restart（不依赖 SQL）
                logger.info(f"  PG 已恢复，继续采集")
            except Exception as e2:
                logger.error(f"  PG 恢复失败: {e2}，后续采集可能全部失败")


    try:
        pg_ctl.reset_to_default()
        pg_ctl.restart()
    except Exception:
        pass

    elapsed = time.time() - t0
    logger.info(f"采集完成: {len(results)} 条，耗时 {int(elapsed)}s → {output_path}")


# ==================== Step 3: 随机采样 knob 配置 ====================

def random_sample_knobs(knob_space_path: str, output_path: str,
                        count: int = 200, strategy: str = "mixed"):
    """随机采样 knob 配置，输出与 generate 相同格式的 JSON

    Args:
        knob_space_path: knob_space.yaml 路径
        output_path: 输出 JSON 路径
        count: 采样数量
        strategy: random / near_default / lhs / mixed
    """
    from cost_model.knob_generator import KnobGenerator
    import random as rand_mod

    knob_space = KnobSpace(knob_space_path)
    gen = KnobGenerator(knob_space)

    logger.info(f"随机采样 {count} 条 knob 配置，策略: {strategy}")

    if strategy == "mixed":
        configs_raw = gen.sample_mixed(count)
    elif strategy == "near_default":
        configs_raw = [gen.sample_near_default() for _ in range(count)]
    elif strategy == "lhs":
        configs_raw = gen.sample_lhs(count)
    else:  # random
        configs_raw = [gen.sample_random() for _ in range(count)]

    # 随机分配 workload 类型
    workloads = ["mixed", "read_only", "high_concurrency", "write_heavy"]

    results = []
    for i, knobs in enumerate(configs_raw):
        results.append({
            "name": f"random_{i:04d}",
            "source": "random_sampled",
            "knobs": knobs,
            "description": f"随机采样配置（策略: {strategy}）",
            "difficulty": 0,
            "workload": rand_mod.choice(workloads),
        })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"已保存 {len(results)} 条随机配置 → {output_path}")


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="故障场景 Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # Step 0: seeds（按瓶颈方向生成）
    sd = subparsers.add_parser("seeds", help="按瓶颈方向生成种子场景描述")
    sd.add_argument("--effects", default="configs/knob_effects.yaml", help="knob 效果知识库")
    sd.add_argument("--knob-space", default="configs/knob_space.yaml", help="knob 搜索空间")
    sd.add_argument("--output", default="datasets/data/scenarios/seeds.json")
    sd.add_argument("--count", type=int, default=100, help="目标种子总数")
    sd.add_argument("--model", default="gpt-5")
    sd.add_argument("--api-key", default=None)
    sd.add_argument("--api-base", default=None)

    # Step 1: generate（LLM 生成 knob 配置）
    gen = subparsers.add_parser("generate", help="LLM 生成可用但有优化空间的 knob 配置")
    gen.add_argument("--seeds", default="datasets/data/scenarios/seeds.json")
    gen.add_argument("--output", default="datasets/data/scenarios/knob_configs_llm.json")
    gen.add_argument("--config", default="configs/knob_space.yaml")
    gen.add_argument("--model", default="gpt-5")
    gen.add_argument("--api-key", default=None)
    gen.add_argument("--api-base", default=None)
    gen.add_argument("--variants", type=int, default=5, help="每个种子生成几个变体")
    gen.add_argument("--workers", type=int, default=5, help="并发线程数")
    gen.add_argument("--cpu", type=int, default=8, help="CPU 核数")
    gen.add_argument("--memory", type=int, default=16, help="内存 GB")
    gen.add_argument("--disk", default="SSD", choices=["SSD", "HDD", "NVMe"], help="磁盘类型")

    # Step 2: random-sample（随机采样 knob 配置）
    rs = subparsers.add_parser("random-sample", help="随机采样 knob 配置（Cost Model 训练数据）")
    rs.add_argument("--knob-space", default="configs/knob_space.yaml")
    rs.add_argument("--output", default="datasets/data/scenarios/knob_configs_random.json")
    rs.add_argument("--count", type=int, default=200, help="采样数量")
    rs.add_argument("--strategy", default="mixed",
                    choices=["random", "near_default", "lhs", "mixed"],
                    help="采样策略：random/near_default/lhs/mixed（默认 mixed=40%%random+40%%near_default+20%%lhs）")

    # Step 3: collect（统一真机采集）
    col = subparsers.add_parser("collect", help="统一真机采集（支持 glob 合并多个 knob_configs_*.json）")
    col.add_argument("--input", default="datasets/data/scenarios/knob_configs_*.json",
                     help="输入 JSON 路径，支持 glob 模式")
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
            effects_path=args.effects,
            knob_space_path=args.knob_space,
            output_path=args.output,
            llm_generate=llm_fn,
            count=args.count,
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

    elif args.command == "random-sample":
        random_sample_knobs(
            knob_space_path=args.knob_space,
            output_path=args.output,
            count=args.count,
            strategy=args.strategy,
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
