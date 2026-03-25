"""
场景采集 Pipeline

Step 1: synthesize — 按维度组合蒸馏让 LLM 系统性生成 knob 配置
Step 2: random-sample — 随机采样 knob 配置（Cost Model 训练数据）
Step 3: collect   — 真机执行 + 采集完整指标（需要 PG）

用法:
    # Step 1: 维度组合蒸馏生成 knob 配置
    python3 -m datasets.synthesis.scenarios.pipeline synthesize \
        --dimensions configs/synthesis_dimensions.yaml \
        --knob-space configs/knob_space.yaml \
        --output datasets/data/scenarios/knob_configs_synth.json \
        --per-cell 5 --workers 5 --model gpt-5

    # Step 2: 随机采样
    python3 -m datasets.synthesis.scenarios.pipeline random-sample \
        --knob-space configs/knob_space.yaml \
        --output datasets/data/scenarios/knob_configs_random.json \
        --count 5000 --strategy mixed

    # Step 3: 真机采集
    python3 -m datasets.synthesis.scenarios.pipeline collect \
        --input 'datasets/data/scenarios/knob_configs_*.json' \
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

# ==================== Step 1: 真机采集 ====================


def collect_scenarios(input_path: str, output_path: str,
                      pg_host: str = "127.0.0.1", pg_port: int = 5432,
                      pg_user: str = "postgres", pg_password: str = "",
                      pg_database: str = "postgres", pg_data_dir: str = None,
                      knob_space_path: str = "configs/knob_space.yaml",
                      start: int = None, end: int = None):
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

    # 分片：--start/--end 截取范围（1-indexed）
    if start is not None or end is not None:
        s = (start or 1) - 1  # 转 0-indexed
        e = end or len(configs)
        configs = configs[s:e]
        logger.info(f"分片: [{s+1}, {e}]，本机负责 {len(configs)} 条")

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

            # 每条保存一次（原子写入，防崩溃时文件损坏）
            import tempfile
            tmp_fd, tmp_path = tempfile.mkstemp(
                suffix='.json', dir=os.path.dirname(output_path) or '.'
            )
            try:
                with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, output_path)
            except Exception:
                os.unlink(tmp_path)
                raise

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


# ==================== Step 4: 维度组合蒸馏 ====================

SYNTHESIZE_PROMPT = """你是 PostgreSQL 性能调优专家。请根据以下场景描述，生成一组完整的 45 个 knob 配置。

## 场景
{description}

## 严重程度
{severity_desc}

## 负载类型
{workload}

## 硬件环境
- CPU: {cpu} 核
- 内存: {memory} GB
- 磁盘: {disk}

## 场景要求的 knob 方向
{knob_hints}

## 可用 knob 列表（必须从这些中选择，值必须在范围内）
{knob_space_summary}

## 要求
1. 输出完整的 45 个 knob 配置（所有 knob 都要有值），而不只是需要修改的那几个
2. 场景要求方向的 knob 按指定方向设置
3. 其他 knob 设置为合理值（不一定是默认值，要考虑与核心 knob 的协同关系）
4. 配置必须可用：数据库能正常启动并跑 benchmark
5. 严重程度 {severity} 意味着：{severity_meaning}
6. 值的格式：memory 类型用 "128MB"/"2GB" 格式，enum 用字符串，integer/float 用数字
7. 输出纯 JSON，不要任何多余的文字

## 输出格式
{{"shared_buffers": "2GB", "work_mem": "4MB", ...所有 45 个 knob...}}"""

SEVERITY_MEANINGS = {
    "mild": "参数轻微偏移，与默认值差距约 ±20%，性能影响不大",
    "moderate": "参数中度偏移，与默认值差距约 ±50%，性能有明显影响",
    "severe": "参数严重偏离，接近极值或偏离 200%+，性能严重受影响",
}


def synthesize_knobs(dimensions_path: str, knob_space_path: str,
                     output_path: str, llm_generate, per_cell: int = 5,
                     workers: int = 1):
    """按维度组合蒸馏生成 knob 配置

    读取 synthesis_dimensions.yaml，按 场景×负载×严重程度 的笛卡尔积
    调用 LLM 生成完整的 knob 配置。
    """
    import yaml
    import threading
    import itertools

    knob_space = KnobSpace(knob_space_path)

    with open(dimensions_path, "r", encoding="utf-8") as f:
        dims = yaml.safe_load(f)

    hardware = dims["hardware"]
    workloads = dims["workloads"]
    severities = dims["severities"]
    scenarios = dims["scenarios"]

    # 构建任务列表
    tasks = []
    for scenario in scenarios:
        severity_list = severities if scenario.get("severity_varies", True) else ["fixed"]
        for workload in workloads:
            for severity in severity_list:
                for v in range(per_cell):
                    tasks.append((scenario, workload, severity, v))

    logger.info(f"场景: {len(scenarios)}, 负载: {len(workloads)}, "
                f"总任务: {len(tasks)}")

    # 加载已有结果（断点续跑）
    results = []
    existing_keys = set()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        existing_keys = {(e["name"], e.get("variant", 0),
                          e.get("workload", ""), e.get("severity", ""))
                         for e in results}
        logger.info(f"已有 {len(results)} 条，断点续跑")

    # 过滤已完成的
    pending = []
    for scenario, workload, severity, v in tasks:
        key = (scenario["name"], v, workload, severity)
        if key not in existing_keys:
            pending.append((scenario, workload, severity, v))

    logger.info(f"待生成: {len(pending)} 条")
    if not pending:
        logger.info("所有任务已完成")
        return

    # 格式化 knob 方向提示
    def _format_knob_hints(scenario):
        lines = []
        for knob_name, info in scenario.get("key_knobs", {}).items():
            if "value" in info:
                lines.append(f"- {knob_name}: 设为 {info['value']}")
            elif "direction" in info:
                direction = info["direction"]
                range_str = info.get("range", "")
                lines.append(f"- {knob_name}: 偏{direction}，参考范围 {range_str}")
        return "\n".join(lines) if lines else "（无特殊要求）"

    lock = threading.Lock()
    counter = {"done": 0, "success": 0, "total": len(pending)}

    def _save():
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def _process_one(args):
        scenario, workload, severity, v = args
        name = scenario["name"]
        label = f"{name}_w{workload}_s{severity}_v{v}"

        try:
            severity_desc = SEVERITY_MEANINGS.get(severity, "固定配置，不区分严重程度")

            prompt = SYNTHESIZE_PROMPT.format(
                description=scenario["description"],
                severity=severity,
                severity_desc=severity_desc,
                severity_meaning=severity_desc,
                workload=workload,
                cpu=hardware["cpu_count"],
                memory=hardware["total_memory_gb"],
                disk=hardware["disk_type"],
                knob_hints=_format_knob_hints(scenario),
                knob_space_summary=knob_space.summarize_for_prompt(),
            )

            raw = llm_generate(prompt)
            text = raw.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:-1])

            knob_config = json.loads(text)
            valid = knob_space.validate(knob_config)
            if not valid:
                with lock:
                    counter["done"] += 1
                    logger.error(f"  [{counter['done']}/{counter['total']}] ❌ {label}: 校验失败")
                return

            result = {
                "name": name,
                "variant": v,
                "source": "llm_generated",
                "difficulty": scenario.get("difficulty", 1),
                "category": scenario.get("category", ""),
                "description": scenario["description"],
                "workload": workload,
                "severity": severity,
                "knobs": valid,
                "hardware_hint": hardware,
            }

            with lock:
                results.append(result)
                counter["done"] += 1
                counter["success"] += 1
                logger.info(f"  [{counter['done']}/{counter['total']}] "
                            f"✅ {label}: {len(valid)} knobs")
                _save()

        except json.JSONDecodeError as e:
            with lock:
                counter["done"] += 1
                logger.error(f"  [{counter['done']}/{counter['total']}] ❌ {label}: JSON 解析失败: {e}")
        except Exception as e:
            with lock:
                counter["done"] += 1
                logger.error(f"  [{counter['done']}/{counter['total']}] ❌ {label}: {e}")

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_one, t) for t in pending]
            for future in as_completed(futures):
                future.result()
    else:
        for t in pending:
            _process_one(t)

    logger.info(f"合成完成: 成功 {counter['success']}/{counter['total']}，"
                f"总计 {len(results)} 条 → {output_path}")


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="场景采集 Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # synthesize（维度组合蒸馏）
    syn = subparsers.add_parser("synthesize", help="按维度组合蒸馏生成 knob 配置（场景×负载×严重程度）")
    syn.add_argument("--dimensions", default="configs/synthesis_dimensions.yaml",
                     help="维度定义文件")
    syn.add_argument("--knob-space", default="configs/knob_space.yaml")
    syn.add_argument("--output", default="datasets/data/scenarios/knob_configs_synth.json")
    syn.add_argument("--per-cell", type=int, default=5, help="每个维度格子生成几条配置")
    syn.add_argument("--model", default="gpt-5")
    syn.add_argument("--api-key", default=None)
    syn.add_argument("--api-base", default=None)
    syn.add_argument("--workers", type=int, default=5, help="并发线程数")

    # random-sample（随机采样 knob 配置）
    rs = subparsers.add_parser("random-sample", help="随机采样 knob 配置（Cost Model 训练数据）")
    rs.add_argument("--knob-space", default="configs/knob_space.yaml")
    rs.add_argument("--output", default="datasets/data/scenarios/knob_configs_random.json")
    rs.add_argument("--count", type=int, default=200, help="采样数量")
    rs.add_argument("--strategy", default="mixed",
                    choices=["random", "near_default", "lhs", "mixed"],
                    help="采样策略：random/near_default/lhs/mixed（默认 mixed=40%%random+40%%near_default+20%%lhs）")

    # collect（统一真机采集）
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
    col.add_argument("--start", type=int, default=None, help="起始编号（1-indexed，含）")
    col.add_argument("--end", type=int, default=None, help="结束编号（1-indexed，含）")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.command == "synthesize":
        from datasets.synthesis.mcts.run_search import create_llm_client
        llm_fn = create_llm_client(
            model=args.model,
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
            api_base=args.api_base or os.environ.get("OPENAI_API_BASE"),
        )
        synthesize_knobs(
            dimensions_path=args.dimensions,
            knob_space_path=args.knob_space,
            output_path=args.output,
            llm_generate=llm_fn,
            per_cell=args.per_cell,
            workers=args.workers,
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
            start=args.start, end=args.end,
        )

    else:
        parser.print_help()

