"""
单 episode 评估逻辑

基于 core.agent.rollout 运行 episode，
从 env tracking 和 tool_history 中提取评估指标。
"""

import logging
from typing import Callable, Dict, Any

logger = logging.getLogger(__name__)


def run_episode(
    env,
    llm_fn: Callable,
    system_prompt: str,
    user_message: str,
    max_turns: int = 10,
    sample_idx: int = 0,
    save_trajectory: bool = False,
) -> Dict[str, Any]:
    """
    运行单个评估 episode，返回结构化指标。

    指标全部从 env 的 tracking 和 tool_history 中提取，
    不做任何 tool_call 解析（那是 env.step 的职责）。
    """
    from core.agent import rollout

    messages, tracking = rollout(
        env=env,
        llm_fn=llm_fn,
        system_prompt=system_prompt,
        user_message=user_message,
        max_turns=max_turns,
    )

    # 从 env tracking 中提取指标
    tool_history = tracking.get("tool_history", [])
    rewards = tracking.get("rewards", [])
    steps = tracking.get("steps_taken", 0)

    # 从 tool_history 统计（不硬编码 tool 名字，按实际调用聚合）
    tool_calls = {}  # tool_name -> count
    knobs_set = set()
    called_predict = False

    for entry in tool_history:
        name = entry.get("tool", "")
        tool_calls[name] = tool_calls.get(name, 0) + 1

        # 从 tool_history 的 args 提取 knob 信息
        if name == "set_knob":
            knob_name = entry.get("args", {}).get("knob_name", "")
            if knob_name:
                knobs_set.add(knob_name)
        if name == "predict_performance":
            called_predict = True

    # 有效/无效动作统计（来自 env tracking）
    valid_count = len(tool_history)  # tool_history 只记录执行成功的
    total_steps = steps

    # 格式合规检查：每个 assistant 轮是否都有 <think> + <tool_call>
    format_ok = True
    for msg in messages:
        if msg["role"] == "assistant":
            c = msg["content"]
            if "<think>" not in c or "</think>" not in c:
                format_ok = False
            if "<tool_call>" not in c or "</tool_call>" not in c:
                format_ok = False

    result = {
        "sample_idx": sample_idx,
        "steps": total_steps,
        "rewards": rewards,
        "total_reward": sum(rewards),
        "called_predict": called_predict,
        "format_pass": format_ok,
        "tool_calls": tool_calls,
        "num_knobs_set": len(knobs_set),
        "knobs_set": list(knobs_set),
    }

    if save_trajectory:
        result["trajectory"] = messages

    return result
