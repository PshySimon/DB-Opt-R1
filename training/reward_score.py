"""
DB 调优 Reward 计算

总分 = format_score + answer_score
- format_score（0 ~ 1.5）：检查对话格式正确性
- answer_score（-1.0 ~ ~1.1）：通过 Cost Model 评估 knob 配置的 TPS 改善

参考 Agent-R1 的 reward_score/compiler_autotuning.py 设计。
"""

import re
import json
import math
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


# ============================================================
# Format Score（格式分）
# ============================================================

def compute_score_format(solution_str: str) -> float:
    """
    检查模型输出的对话格式正确性。

    评分项：
    - 基础结构（start/end 标签）
    - 轮次交替（assistant → user → assistant ...）
    - <think> 推理标签
    - <tool_call> 工具调用格式

    Returns:
        float: 0 ~ 1.5（完美格式 = 1.5）
    """
    if not solution_str or not isinstance(solution_str, str):
        return 0.0

    score = 0.0
    is_perfect = True

    # --- 基础结构检查 ---
    if "<error>" in solution_str:
        is_perfect = False
    else:
        score += 0.10

    has_correct_start = solution_str.startswith("<|im_start|>assistant")
    has_correct_end = solution_str.strip().endswith("<|im_end|>")

    if has_correct_start:
        score += 0.08
    else:
        is_perfect = False
    if has_correct_end:
        score += 0.08
    else:
        is_perfect = False

    # --- 分割轮次 ---
    turns_raw = re.split(r'(<\|im_start\|>)', solution_str)
    turns = []
    if len(turns_raw) > 1:
        for i in range(1, len(turns_raw), 2):
            if i + 1 < len(turns_raw):
                turns.append(turns_raw[i] + turns_raw[i + 1])

    if not turns:
        is_perfect = False
        return min(score, 1.49)

    score += 0.04

    # start/end 数量匹配
    start_count = len(turns)
    end_count = solution_str.count("<|im_end|>")
    if start_count == end_count:
        score += 0.10
    else:
        is_perfect = False

    # --- 逐轮检查 ---
    expected_role = 'assistant'

    for i, turn_text in enumerate(turns):
        turn_text_stripped = turn_text.strip()

        # 结束标签
        if turn_text_stripped.endswith("<|im_end|>"):
            score += 0.01
        else:
            is_perfect = False

        # 提取角色和内容
        content_match = re.match(
            r"<\|im_start\|>(.*?)<\|im_end\|>?$",
            turn_text_stripped, re.DOTALL
        )
        if not content_match:
            is_perfect = False
            continue

        turn_content = content_match.group(1).strip()
        actual_role = None
        if turn_content.startswith("assistant"):
            actual_role = 'assistant'
        elif turn_content.startswith("user"):
            actual_role = 'user'

        if actual_role:
            score += 0.01
            if actual_role == expected_role:
                score += 0.01
            else:
                is_perfect = False
        else:
            is_perfect = False
            continue

        turn_payload = turn_content[len(actual_role):].strip()

        # assistant 轮次检查
        if actual_role == 'assistant':
            # <think> 标签
            if "<think>" in turn_payload and "</think>" in turn_payload:
                score += 0.02
            else:
                is_perfect = False

            # <tool_call> 或 <answer> 标签
            if "<tool_call>" in turn_payload:
                score += 0.03
                # 检查 JSON 格式
                tool_match = re.search(
                    r'<tool_call>(.*?)</tool_call>',
                    turn_payload, re.DOTALL
                )
                if tool_match:
                    try:
                        tool_data = json.loads(tool_match.group(1).strip())
                        if "name" in tool_data:
                            score += 0.02
                    except json.JSONDecodeError:
                        is_perfect = False

        # user 轮次检查
        elif actual_role == 'user':
            if "<tool_response>" in turn_payload:
                score += 0.02

        # 交替角色
        expected_role = 'user' if actual_role == 'assistant' else 'assistant'

    # --- 最终评分 ---
    if is_perfect:
        return 1.5
    else:
        return min(score, 1.49)


# ============================================================
# Answer Score（任务分）
# ============================================================

def extract_final_knobs(solution_str: str) -> Optional[Dict[str, str]]:
    """
    从模型输出的轨迹中提取最终设置的 knob 配置。

    扫描所有 <tool_call> 中调用 set_knob 的记录，
    返回最终（最后一次设置的）knob 值。
    """
    if not solution_str:
        return None

    knobs = {}
    # 找所有 tool_call
    tool_calls = re.findall(
        r'<tool_call>(.*?)</tool_call>',
        solution_str, re.DOTALL
    )

    for tc in tool_calls:
        try:
            data = json.loads(tc.strip())
            if data.get("name") == "set_knob":
                args = data.get("arguments", {})
                knob_name = args.get("knob_name")
                value = args.get("value")
                if knob_name and value is not None:
                    knobs[knob_name] = str(value)
        except (json.JSONDecodeError, AttributeError):
            continue

    return knobs if knobs else None


def compute_score_answer(
    solution_str: str,
    ground_truth: dict,
    cost_model=None,
) -> float:
    """
    任务完成度评分：提取 knob → Cost Model 预测 TPS → 计算改善比例。

    Args:
        solution_str: 模型输出的完整轨迹
        ground_truth: 包含 baseline_tps、hardware 等场景信息
        cost_model: CostModel 实例（predict 接口）

    Returns:
        float: -1.0 ~ ~1.1
        - 正值：TPS 有改善（log 压缩）
        - 负值：TPS 劣化（clip 到 -1）
        - 0：无法提取 knob 或无 cost_model
    """
    if cost_model is None:
        return 0.0

    # 提取最终 knob 配置
    knobs = extract_final_knobs(solution_str)
    if not knobs:
        return 0.0

    # Cost Model 预测
    try:
        hardware = ground_truth.get("hardware", {})
        
        from core.db.knob_space import KnobSpace
        ks = KnobSpace("configs/knob_space.yaml")
        default_knobs = ks.get_default_config()
        baseline_tps = cost_model.predict(default_knobs, hardware)

        if baseline_tps <= 0:
            return 0.0

        predicted_tps = cost_model.predict(knobs, hardware)

        # 计算改善比例，cap 到 [0, 2.0]（200%）
        improvement = (predicted_tps - baseline_tps) / baseline_tps
        improvement = min(2.0, max(0.0, improvement))

        # log 压缩，避免大改善爆炸
        score = math.log(1 + improvement)

        return score

    except Exception as e:
        logger.warning(f"Cost Model 预测失败: {e}")
        return 0.0


def compute_score_format_answer(
    solution_str: str,
    ground_truth: dict,
    cost_model=None,
) -> float:
    """
    总分 = format_score + answer_score

    Returns:
        float: ~(-1.0) ~ ~2.6
    """
    format_score = compute_score_format(solution_str)
    answer_score = compute_score_answer(solution_str, ground_truth, cost_model)
    return format_score + answer_score
