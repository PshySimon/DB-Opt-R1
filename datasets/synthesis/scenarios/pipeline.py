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


def generate_questions_for_state(state, n: int, llm_fn) -> list:
    """一次 LLM 调用为场景生成 n 条风格各异的 question。

    接受 ScenarioState 实例或 dict（synthesize 阶段的原始 knob_config）。
    prompt 素材：description, workload, severity。
    要求 LLM 返回 JSON："{"questions": ["...", "...", ...]}"
    失败直接抛异常，不做 fallback。
    """
    import re as _re
    import random
    import json as _json

    if isinstance(state, dict):
        description = state.get("description", "")
        wl_raw = state.get("workload", "mixed")
        wl = wl_raw if isinstance(wl_raw, str) else wl_raw.get("type", "mixed")
        severity = state.get("severity", "medium")
        knobs = state.get("knobs", {})
    else:
        description = getattr(state, "description", "") or ""
        wl_obj = getattr(state, "workload", {}) or {}
        wl = wl_obj.get("type", "mixed") if isinstance(wl_obj, dict) else str(wl_obj)
        severity = ""
        knobs = getattr(state, "knobs", {}) or {}

    wl_desc = {
        "read_only": "主要读多写少的业务",
        "write_heavy": "写操作频繁的业务",
        "high_concurrency": "高并发业务",
        "mixed": "读写混合业务",
    }.get(wl, "读写混合业务")

    severity_hint = ""
    if severity in ("high", "critical"):
        severity_hint = "，问题比较严重"
    elif severity == "medium":
        severity_hint = "，性能有明显下降"

    personas = [
        "业务开发同学，不太懂数据库底层，只知道系统有问题",
        "公司 DBA，说话直接，知道是 PG 配置问题",
        "运营同学，只关心业务受影响，不懂技术细节",
        "刚接手系统的后端工程师，有些不确定",
        "产品经理，关注用户体验下降，不懂技术",
    ]

    # 构建 knob 偏差描述（给 LLM 内部参考，推断用户会感受到的症状）
    knob_hint = ""
    if knobs:
        knob_lines = ["  - {}: {}".format(k, v) for k, v in knobs.items()]
        knob_hint = (
            "\n当前有问题的数据库配置（仅供你推断症状，绝对不要在问题中出现这些参数名）：\n"
            + "\n".join(knob_lines) + "\n"
        )

    personas_str = ", ".join(personas)
    json_fmt = '{"questions": ["...", "...", ...]}'

    prompt = (
        "你需要模拟 {} 个不同身份的用户，分别向 AI 数据库调优助手发出一句求助或指令。\n\n"
        "背景：这是一个{}的系统{}。\n"
        "问题描述（内部参考，不要照搬原文）：{}\n"
        "{}\n"
        "可选人物背景（随机分配，不同问题用不同身份）：{}\n\n"
        "示例（仅展示风格，不要照抄内容）：\n"
        "- 「我们系统最近怎么越来越卡，下单那块尤其明显，你帮看看数据库是不是有问题」\n"
        "- 「帮我把数据库调一下，写操作现在慢得不行」\n"
        "- 「报表查询老半天出不来，用户投诉了，麻烦帮看看怎么优化」\n"
        "- 「并发一高就开始堆积，能不能帮我排查一下瓶颈在哪」\n"
        "- 「数据库这几天感觉有点问题，你帮我调一调」\n\n"
        "要求：\n"
        "1. 生成恰好 {} 条问题，每条使用不同身份和措辞，风格各异\n"
        "2. 完全以第一人称口吻，贴近真实对话\n"
        "3. 根据上面的配置偏差推断用户实际会感受到的症状（如卡顿、超时、磁盘响、内存不够等），体现在问题中\n"
        "4. 绝对不能出现任何技术指标名称（如 buffer_hit_rate、seq_scan、TPS、wait_event、knob 名称等）\n"
        "5. 不能出现任何具体数字\n"
        "6. 每条不超过 60 字\n"
        "7. 输出严格的 JSON，格式为：{}（恰好 {} 条）\n\n"
        "JSON:"
    ).format(n, wl_desc, severity_hint, description, knob_hint, personas_str, n, json_fmt, n)

    raw = llm_fn(prompt)
    m = _re.search(r'\{.*?"questions"\s*:\s*(\[.*?\])\s*\}', raw, _re.DOTALL)
    if m:
        questions = _json.loads(m.group(1))
        if isinstance(questions, list) and len(questions) >= n:
            return [str(q).strip() for q in questions[:n]]
    parsed = _json.loads(raw)
    questions = parsed.get("questions", [])
    if isinstance(questions, list) and len(questions) >= n:
        return [str(q).strip() for q in questions[:n]]
    raise ValueError(f"LLM 未返回足够的 questions（需要 {n} 条），原始输出: {raw[:300]}")


# ==================== Step 1: 真机采集 ====================


def _run_io_benchmark() -> dict:
    """运行一次 IO 基准测试，返回硬件 IO 特征字典。
    整个 collect 任务启动时采集一次，写入所有记录。
    依赖 fio（需提前安装: apt install fio --fix-missing）。
    """
    import subprocess, re, shutil, tempfile
    result = {}

    fio_bin = shutil.which("fio") or os.path.expanduser("~/.local/bin/fio")

    with tempfile.TemporaryDirectory() as tmpdir:
        fio_file = os.path.join(tmpdir, "fio_test")

        # -- 1. 顺序写 (fio, 512MB, 30s) ---------------------------------
        try:
            if os.path.exists(fio_bin):
                out = subprocess.check_output([
                    fio_bin, "--name=seq_write", "--rw=write", "--bs=1M",
                    "--size=512M", "--numjobs=1", "--runtime=30", "--time_based",
                    "--group_reporting", "--output-format=terse",
                    f"--filename={fio_file}",
                ], stderr=subprocess.DEVNULL, timeout=60).decode()
                fields = out.split(";")
                if len(fields) > 50:
                    result["seq_write_bw_fio_mbps"] = round(float(fields[48]) / 1024, 1)  # KiB/s → MiB/s
                    result["seq_write_p99_lat_us"] = int(float(fields[56]))               # p99 clat (us)
        except Exception as e:
            logger.warning(f"IO 基准(顺序写)失败: {e}")

        # -- 2. 随机读 4K (fio, 512MB, 30s) -------------------------------
        try:
            if os.path.exists(fio_bin):
                out = subprocess.check_output([
                    fio_bin, "--name=rand_read", "--rw=randread", "--bs=4k",
                    "--size=512M", "--numjobs=4", "--runtime=30", "--time_based",
                    "--group_reporting", "--output-format=terse",
                    f"--filename={fio_file}",
                ], stderr=subprocess.DEVNULL, timeout=60).decode()
                fields = out.split(";")
                if len(fields) > 10:
                    result["rand_read_iops"] = int(float(fields[7]))      # read IOPS
                    result["rand_read_mbps"] = round(float(fields[6]) / 1024, 1)  # KiB/s → MiB/s
        except Exception as e:
            logger.warning(f"IO 基准(随机读)失败: {e}")

    # -- 3. 顺序写 (dd, 256MB, fdatasync) ---------------------------------
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dd") as tf:
            dd_file = tf.name
        t0 = time.time()
        subprocess.check_call(
            f"dd if=/dev/zero of={dd_file} bs=1M count=256 conv=fdatasync",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60
        )
        elapsed = time.time() - t0
        result["seq_write_mbps"] = round(256 / elapsed, 1)

        # -- 4. 顺序读 (dd, 无缓存) ----------------------------------------
        try:
            subprocess.call("echo 3 > /proc/sys/vm/drop_caches", shell=True,
                            stderr=subprocess.DEVNULL)
        except Exception:
            pass
        t0 = time.time()
        subprocess.check_call(
            f"dd if={dd_file} of=/dev/null bs=1M",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60
        )
        result["seq_read_mbps"] = round(256 / (time.time() - t0), 1)
        os.unlink(dd_file)
    except Exception as e:
        logger.warning(f"IO 基准(dd 顺序读写)失败: {e}")

    # -- 5. 内存带宽 (dd /dev/zero → /dev/null) ---------------------------
    try:
        t0 = time.time()
        subprocess.check_call(
            "dd if=/dev/zero of=/dev/null bs=1M count=4096",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30
        )
        result["mem_bw_gbps"] = round(4 / (time.time() - t0), 2)  # 4 GB / s
    except Exception as e:
        logger.warning(f"IO 基准(内存带宽)失败: {e}")

    logger.info(f"IO 基准采集完成: {result}")
    return result


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

    # 采集任务开始前，先做一次 IO 基准测试（只跑一次，写入所有记录）
    logger.info("开始 IO 基准测试（约 2 分钟）...")
    io_benchmarks = _run_io_benchmark()

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

            # 将 IO 基准数据注入 hardware 字段
            hw_with_io = {k: v for k, v in hardware.items() if v is not None}
            hw_with_io.update(io_benchmarks)

            from dataclasses import asdict
            state = ScenarioState(
                name=config["name"],
                variant=config.get("variant", 0),
                source=config.get("source", "llm_generated"),
                difficulty=config.get("difficulty", 1),
                description=config.get("description", ""),
                hardware=hw_with_io,
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


# ==================== Step 5: 贝叶斯优化搜索好配置 ====================

def bo_search(knob_space_path: str, collected_path: str, output_path: str,
              pg_host: str, pg_port: int, pg_user: str, pg_password: str,
              pg_database: str, pg_data_dir: str,
              rounds: int = 30, workloads: list = None,
              n_perturb: int = 50):
    """贝叶斯优化搜索好配置

    对每种 workload 独立跑 BO，用已有 collected 数据热启动。
    搜索轨迹 + 围绕 top 配置的扰动变体一起输出。
    """
    from skopt import Optimizer
    from skopt.space import Real, Integer, Categorical
    import numpy as np
    import random as _random

    knob_space = KnobSpace(knob_space_path)
    from core.db.knob_space import parse_memory, format_memory

    if workloads is None:
        workloads = ["read_only", "write_heavy", "mixed", "high_concurrency"]

    # ---- 构建 skopt 搜索空间 ----
    dimensions = []
    dim_names = []
    knob_types = {}  # name → type info

    for knob_name, info in knob_space.knobs.items():
        knob_type = info["type"]
        knob_types[knob_name] = info

        if knob_type == "memory":
            lo = parse_memory(str(info["min"]))
            hi = parse_memory(str(info["max"]))
            dimensions.append(Real(float(lo), float(hi), name=knob_name))
        elif knob_type == "integer":
            dimensions.append(Integer(int(info["min"]), int(info["max"]), name=knob_name))
        elif knob_type == "float":
            dimensions.append(Real(float(info["min"]), float(info["max"]), name=knob_name))
        elif knob_type == "enum":
            dimensions.append(Categorical(info["values"], name=knob_name))

        dim_names.append(knob_name)

    def _knobs_to_x(knobs_dict: dict) -> list:
        """knob dict → skopt 向量"""
        x = []
        for name in dim_names:
            info = knob_types[name]
            val = knobs_dict.get(name, info["default"])
            if info["type"] == "memory":
                x.append(float(parse_memory(str(val))))
            elif info["type"] == "integer":
                x.append(int(val))
            elif info["type"] == "float":
                x.append(float(val))
            elif info["type"] == "enum":
                x.append(str(val))
        return x

    def _x_to_knobs(x: list) -> dict:
        """skopt 向量 → knob dict（格式化为 PG 可接受的值）"""
        knobs = {}
        for i, name in enumerate(dim_names):
            info = knob_types[name]
            val = x[i]
            if info["type"] == "memory":
                knobs[name] = format_memory(int(round(val)))
            elif info["type"] == "integer":
                knobs[name] = int(round(val))
            elif info["type"] == "float":
                knobs[name] = round(float(val), 4)
            elif info["type"] == "enum":
                knobs[name] = str(val)
        return knobs

    # ---- 加载已有数据做热启动 ----
    existing_data = {}  # workload → [(x, tps)]
    if collected_path and os.path.exists(collected_path):
        import glob as glob_mod
        files = sorted(glob_mod.glob(collected_path))
        if not files:
            files = [collected_path]
        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                if not isinstance(item, dict):
                    continue
                wl = item.get("workload", {})
                if isinstance(wl, dict):
                    wl_type = wl.get("type", "mixed")
                    tps = wl.get("tps_current", 0)
                else:
                    wl_type = "mixed"
                    tps = 0
                if tps <= 0:
                    continue
                knobs = item.get("knobs", {})
                if not knobs:
                    continue
                try:
                    x = _knobs_to_x(knobs)
                    existing_data.setdefault(wl_type, []).append((x, tps))
                except Exception:
                    continue
        for wl, pairs in existing_data.items():
            logger.info(f"热启动 [{wl}]: {len(pairs)} 条已有数据")

    # ---- PG 连接 ----
    pg_ctl = PGConfigurator(
        pg_host=pg_host, pg_port=pg_port,
        pg_user=pg_user, pg_password=pg_password,
        pg_database=pg_database, pg_data_dir=pg_data_dir,
    )

    # ---- 加载已有 BO 结果（断点续跑）----
    results = []
    existing_bo_keys = set()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        existing_bo_keys = {(r.get("name", ""), r.get("variant", 0)) for r in results}
        logger.info(f"已有 BO 结果 {len(results)} 条")

    def _save():
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

    def _benchmark_one(knobs: dict, workload: str, label: str) -> float:
        """对一组 knob 配置跑 benchmark，返回 TPS"""
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
            tps = perf.get("tps", 0)
            logger.info(f"  {label}: TPS={tps:.1f}")
            return float(tps)
        except Exception as e:
            logger.error(f"  {label}: benchmark 失败: {e}")
            try:
                pg_ctl.safe_restart()
            except Exception:
                pass
            return 0.0

    # ---- 对每种 workload 跑 BO ----
    for wl in workloads:
        logger.info(f"\n{'='*50}")
        logger.info(f"BO 搜索: workload={wl}, rounds={rounds}")
        logger.info(f"{'='*50}")

        # 检查断点：已完成多少轮
        done_rounds = sum(1 for r in results
                         if r.get("name", "").startswith(f"bo_{wl}_round_"))
        if done_rounds >= rounds:
            logger.info(f"  [{wl}] 已完成 {done_rounds} 轮，跳过")
            continue

        optimizer = Optimizer(
            dimensions=dimensions,
            base_estimator="GP",
            acq_func="EI",
            n_initial_points=max(5, 10 - len(existing_data.get(wl, []))),
            random_state=42,
        )

        # 热启动：用已有数据 tell optimizer
        warm_data = existing_data.get(wl, [])
        if warm_data:
            xs = [pair[0] for pair in warm_data]
            ys = [-pair[1] for pair in warm_data]  # skopt 是最小化，取负
            try:
                optimizer.tell(xs, ys)
                logger.info(f"  热启动 {len(warm_data)} 条")
            except Exception as e:
                logger.warning(f"  热启动失败（跳过）: {e}")

        best_tps = 0.0
        best_knobs = None

        for r in range(done_rounds, rounds):
            name = f"bo_{wl}_round_{r}"
            if (name, 0) in existing_bo_keys:
                continue

            x = optimizer.ask()
            knobs = _x_to_knobs(x)

            label = f"[{wl}] round {r+1}/{rounds}"
            tps = _benchmark_one(knobs, wl, label)

            if tps > 0:
                optimizer.tell(x, -tps)

                result = {
                    "name": name,
                    "variant": 0,
                    "source": "bo_search",
                    "category": "bo",
                    "description": f"BO 搜索 round {r} for {wl}",
                    "workload": wl,
                    "knobs": knobs,
                    "tps": tps,
                }
                results.append(result)
                _save()

                if tps > best_tps:
                    best_tps = tps
                    best_knobs = knobs.copy()

            logger.info(f"  round {r+1}: TPS={tps:.1f}, best={best_tps:.1f}")

        if best_knobs:
            logger.info(f"  [{wl}] 最优 TPS={best_tps:.1f}")

            # ---- 围绕 top 配置生成扰动变体 ----
            logger.info(f"  [{wl}] 生成 {n_perturb} 条扰动变体...")
            for p in range(n_perturb):
                name = f"bo_{wl}_perturb_{p}"
                if (name, 0) in existing_bo_keys:
                    continue

                perturbed = {}
                for knob_name, val in best_knobs.items():
                    info = knob_types.get(knob_name, {})
                    if info.get("type") == "memory":
                        val_kb = parse_memory(str(val))
                        lo = parse_memory(str(info["min"]))
                        hi = parse_memory(str(info["max"]))
                        noise = val_kb * _random.uniform(-0.2, 0.2)
                        new_val = max(lo, min(hi, int(val_kb + noise)))
                        perturbed[knob_name] = format_memory(new_val)
                    elif info.get("type") == "integer":
                        lo, hi = int(info["min"]), int(info["max"])
                        noise = int(val) * _random.uniform(-0.2, 0.2)
                        new_val = max(lo, min(hi, int(round(int(val) + noise))))
                        perturbed[knob_name] = new_val
                    elif info.get("type") == "float":
                        lo, hi = float(info["min"]), float(info["max"])
                        noise = float(val) * _random.uniform(-0.2, 0.2)
                        new_val = max(lo, min(hi, float(val) + noise))
                        perturbed[knob_name] = round(new_val, 4)
                    elif info.get("type") == "enum":
                        # 小概率随机换
                        if _random.random() < 0.1:
                            perturbed[knob_name] = _random.choice(info["values"])
                        else:
                            perturbed[knob_name] = val
                    else:
                        perturbed[knob_name] = val

                results.append({
                    "name": name,
                    "variant": 0,
                    "source": "bo_perturb",
                    "category": "bo",
                    "description": f"围绕 {wl} 最优配置的扰动变体",
                    "workload": wl,
                    "knobs": perturbed,
                })

            _save()
            logger.info(f"  [{wl}] 扰动变体已保存")

    logger.info(f"\nBO 搜索完成: {len(results)} 条 → {output_path}")


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
    syn.add_argument("--providers-config", default=None,
                     help="多中转站 JSON 配置文件路径，传入后忽略单节点 --api-key/--api-base")
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

    # bo-search（贝叶斯优化搜索好配置）
    bo = subparsers.add_parser("bo-search", help="贝叶斯优化搜索好配置（需要 PG）")
    bo.add_argument("--knob-space", default="configs/knob_space.yaml")
    bo.add_argument("--collected", default="datasets/data/scenarios/collected*.json",
                    help="已有 collected 数据（glob），用于热启动")
    bo.add_argument("--output", default="datasets/data/scenarios/collected_bo.json")
    bo.add_argument("--rounds", type=int, default=30, help="每种负载的 BO 轮数")
    bo.add_argument("--workloads", default="read_only,write_heavy,mixed,high_concurrency",
                    help="逗号分隔的负载类型列表")
    bo.add_argument("--n-perturb", type=int, default=50, help="围绕最优配置的扰动变体数")
    bo.add_argument("--host", default="127.0.0.1")
    bo.add_argument("--port", type=int, default=5432)
    bo.add_argument("--user", default="postgres")
    bo.add_argument("--password", default="")
    bo.add_argument("--database", default="postgres")
    bo.add_argument("--pg-data-dir", default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.command == "synthesize":
        from core.llm.multi_client import MultiProviderLLMClient
        llm_client = MultiProviderLLMClient(
            target_model=args.model,
            providers_config=args.providers_config,
            single_api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
            single_api_base=args.api_base or os.environ.get("OPENAI_API_BASE"),
        )
        synthesize_knobs(
            dimensions_path=args.dimensions,
            knob_space_path=args.knob_space,
            output_path=args.output,
            llm_generate=llm_client.generate,
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

    elif args.command == "bo-search":
        bo_search(
            knob_space_path=args.knob_space,
            collected_path=args.collected,
            output_path=args.output,
            pg_host=args.host, pg_port=args.port,
            pg_user=args.user, pg_password=args.password,
            pg_database=args.database, pg_data_dir=args.pg_data_dir,
            rounds=args.rounds,
            workloads=args.workloads.split(","),
            n_perturb=args.n_perturb,
        )

    else:
        parser.print_help()


