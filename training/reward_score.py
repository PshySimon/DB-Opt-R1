"""
DB 调优 Reward 计算

总分 = 0.5 * format_score + 0.5 * answer_score + termination_adj
- format_score（0 ~ 1.5）：检查对话格式正确性 → 加权后 [0, 0.75]
- answer_score（-2.5 ~ 10.0）：通过 Cost Model 评估 knob 配置的 TPS 改善 → 加权后 [-1.25, 5.0]
- answer 信号主导 GRPO 组内方差（~87%），format 保底防退化（~13%）

参考 Compiler-R1 的 reward 设计。
"""

import re
import json
from typing import Optional, Dict, Any, List


TERMINATION_REASON_ADJUSTMENTS = {
    "repeated_tool_call": -0.1,
    "invalid_tool_call": -0.1,
    "tool_execution_error": -0.1,
    "max_turns_reached": -0.05,
}


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
                knob_blob = args.get("knobs")
                if knob_blob:
                    if isinstance(knob_blob, str):
                        parsed_knobs = json.loads(knob_blob)
                    elif isinstance(knob_blob, dict):
                        parsed_knobs = knob_blob
                    else:
                        parsed_knobs = {}
                    for knob_name, value in parsed_knobs.items():
                        if knob_name and value is not None:
                            knobs[knob_name] = str(value)
                    continue

                knob_name = args.get("knob_name")
                value = args.get("value")
                if knob_name and value is not None:
                    knobs[knob_name] = str(value)
        except (json.JSONDecodeError, AttributeError):
            continue

    return knobs if knobs else None


def _parse_set_knobs(args: Any) -> Dict[str, str]:
    if not isinstance(args, dict):
        return {}
    knob_blob = args.get("knobs")
    if knob_blob:
        if isinstance(knob_blob, str):
            try:
                parsed_knobs = json.loads(knob_blob)
            except json.JSONDecodeError:
                parsed_knobs = {}
        elif isinstance(knob_blob, dict):
            parsed_knobs = knob_blob
        else:
            parsed_knobs = {}
        return {
            str(knob_name): str(value)
            for knob_name, value in parsed_knobs.items()
            if knob_name and value is not None
        }

    knob_name = args.get("knob_name")
    value = args.get("value")
    if knob_name and value is not None:
        return {str(knob_name): str(value)}
    return {}


def _payload_from_tool_response(content: str) -> Optional[dict]:
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(
        r"<tool_response>\n?(.*?)\n?</tool_response>",
        content,
        re.DOTALL,
    )
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _iter_tool_events_from_messages(messages: List[dict]):
    for msg in messages or []:
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "assistant":
            for match in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
                try:
                    data = json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    yield "call", data
        elif role in {"tool", "user"}:
            payload = _payload_from_tool_response(content)
            if payload is not None:
                yield "response", payload


def _iter_tool_events_from_solution(solution_str: str):
    if not solution_str:
        return
    pattern = re.compile(
        r"<tool_call>(.*?)</tool_call>|<tool_response>\n?(.*?)\n?</tool_response>",
        re.DOTALL,
    )
    for match in pattern.finditer(solution_str):
        if match.group(1) is not None:
            try:
                data = json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                yield "call", data
        else:
            try:
                payload = json.loads(match.group(2))
            except Exception:
                continue
            if isinstance(payload, dict):
                yield "response", payload


def _is_predict_payload(payload: dict) -> bool:
    return {"predicted_tps", "baseline_tps", "improvement_pct"} <= payload.keys()


def _best_predict_key(payload: dict) -> tuple[float, float]:
    try:
        predicted_tps = float(payload.get("predicted_tps") or 0.0)
    except Exception:
        predicted_tps = 0.0
    try:
        improvement_pct = float(payload.get("improvement_pct") or 0.0)
    except Exception:
        improvement_pct = 0.0
    return predicted_tps, improvement_pct


def _extract_best_predict_from_events(events) -> Optional[Dict[str, Any]]:
    current_knobs: Dict[str, str] = {}
    best: Optional[Dict[str, Any]] = None
    best_key: Optional[tuple[float, float]] = None

    for event_type, data in events:
        if event_type == "call" and data.get("name") == "set_knob":
            current_knobs.update(_parse_set_knobs(data.get("arguments", {})))
            continue

        if event_type != "response" or not _is_predict_payload(data):
            continue

        key = _best_predict_key(data)
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "knobs": dict(current_knobs),
                "payload": dict(data),
            }

    return best


def extract_best_predict_from_messages(messages: List[dict]) -> Optional[Dict[str, Any]]:
    """Return the knob snapshot paired with the best predict_performance result."""
    return _extract_best_predict_from_events(_iter_tool_events_from_messages(messages))


def extract_best_predict_from_solution(solution_str: str) -> Optional[Dict[str, Any]]:
    """Return the knob snapshot paired with the best predict_performance result."""
    return _extract_best_predict_from_events(_iter_tool_events_from_solution(solution_str))


def extract_best_predict_knobs(solution_str: str) -> Optional[Dict[str, str]]:
    best = extract_best_predict_from_solution(solution_str)
    if not best:
        return None
    knobs = best.get("knobs") or {}
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
        cost_model: 保留为兼容旧调用；当前 reward 直接复用轨迹里的 predict_performance 结果

    Returns:
        float: -2.5 ~ 10.0
        - 正值：TPS 有改善（线性放大 ×5）
        - 0：无法提取 best predict 结果，或没有对应 knob
    """
    best = extract_best_predict_from_solution(solution_str)
    if not best or not (best.get("knobs") or {}):
        return 0.0

    payload = best.get("payload") or {}
    try:
        improvement_pct = float(payload.get("improvement_pct") or 0.0)
    except Exception:
        return 0.0

    improvement = min(200.0, max(0.0, improvement_pct)) / 100.0
    return improvement * 5


def compute_termination_adjustment(termination_reason: Optional[str]) -> float:
    if termination_reason is None:
        return 0.0
    return TERMINATION_REASON_ADJUSTMENTS.get(str(termination_reason), 0.0)


def compute_score_format_answer(
    solution_str: str,
    ground_truth: dict,
    cost_model=None,
    termination_reason: Optional[str] = None,
) -> float:
    """
    总分 = 0.5 * format_score + 0.5 * answer_score + termination_adj

    format 占比 ~13%（[0, 0.75]），answer 占比 ~87%（[-2.5, 5.0]）。
    参考 Compiler-R1 的加权设计，确保 answer 信号主导 GRPO 组内方差。

    Returns:
        float: ~(-2.6) ~ ~5.75
    """
    format_score = compute_score_format(solution_str)
    answer_score = compute_score_answer(solution_str, ground_truth, cost_model)
    return 0.5 * format_score + 0.5 * answer_score + compute_termination_adjustment(termination_reason)
