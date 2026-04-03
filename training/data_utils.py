"""
共享数据加载工具

被 trl 和 verl 后端共同使用。
"""

import json
import glob
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是 PostgreSQL 数据库调优专家。你的目标是通过调整数据库配置参数来最大化性能（TPS）。

## 输出格式

每次回复必须严格遵循以下格式：先用 <think>...</think> 分析推理，再用 <tool_call>...</tool_call> 调用一个工具。

重要规则：
1. 每次只能调用一个工具
2. 必须先 <think> 分析，再 <tool_call> 调用
3. <think> 中要解释你观察到了什么、为什么这样做

## 工作流程

1. 先观察硬件环境（get_hardware_info）
2. 查看关键配置（get_current_config）和运行指标（get_db_metrics）
3. 分析瓶颈，用 set_knob 设置合理的参数
4. 如果修改了 shared_buffers 等 static 参数，调用 restart_pg
5. 用 predict_performance 验证效果"""


def load_sft_data(paths: List[str]) -> List[dict]:
    """加载 JSONL 轨迹文件，返回 messages list。

    Args:
        paths: JSONL 文件路径列表（支持 glob）

    Returns:
        [{"messages": [...], "reward": float, ...}, ...]
    """
    resolved = []
    for p in paths:
        resolved.extend(glob.glob(p))

    records = []
    for fpath in resolved:
        logger.info(f"加载 {fpath}")
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    logger.info(f"总计 {len(records)} 条轨迹")
    return records


def load_grpo_prompts(scenario_dir: str, system_prompt: str = None) -> List[dict]:
    """从场景数据构造 GRPO prompt。

    Args:
        scenario_dir: 场景 JSON 文件/目录
        system_prompt: 自定义 system prompt

    Returns:
        [{"prompt": [messages], "ground_truth": {...}}, ...]
    """
    import os
    sys_prompt = system_prompt or SYSTEM_PROMPT

    # 加载场景
    scenarios = []
    if os.path.isfile(scenario_dir):
        with open(scenario_dir) as f:
            scenarios = json.load(f)
    elif os.path.isdir(scenario_dir):
        for fpath in sorted(glob.glob(os.path.join(scenario_dir, "collected*.json"))):
            with open(fpath) as f:
                scenarios.extend(json.load(f))

    # 只取 llm_generated
    scenarios = [s for s in scenarios if s.get("source", "llm_generated") == "llm_generated"]

    prompts = []
    for s in scenarios:
        hw = s.get("hardware", {})
        user_msg = (
            f"请优化这个 PostgreSQL 数据库的性能。\n\n"
            f"硬件环境：CPU {hw.get('cpu_count', '?')} 核，"
            f"内存 {hw.get('total_memory_gb', '?')} GB，"
            f"存储 {hw.get('storage_type', '?')}"
        )
        prompts.append({
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            "ground_truth": {
                "hardware": hw,
                "knobs": s.get("knobs", {}),
            },
        })

    logger.info(f"生成 {len(prompts)} 条 GRPO prompt")
    return prompts
