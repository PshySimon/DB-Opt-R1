"""
Agent Rollout — 共享的 LLM ↔ 环境交互循环

MCTS、评估、GRPO 训练都复用这个函数。
rollout 本身不知道任何具体 tool 名字，不做业务指标统计。
"""

import logging
from typing import Callable, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


def rollout(
    env,
    llm_fn: Callable,
    system_prompt: str,
    user_message: str,
    max_turns: int = 10,
    temperature: float = 0.3,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    运行一个完整的 agent episode。

    Args:
        env: ToolEnv 实例（调用前必须已 reset）
        llm_fn: (messages: list[dict], temperature: float) -> str
        system_prompt: 系统提示词（不含工具描述，会自动拼接）
        user_message: 用户问题
        max_turns: 最大交互轮数
        temperature: LLM 采样温度

    Returns:
        (messages, tracking)
        - messages: 完整对话历史 [system, user, assistant, user(tool_response), ...]
        - tracking: env.get_tracking_variables() 的结果
    """
    # 拼接 system prompt + 工具描述
    tools_desc = env.tools_format_func()
    full_system = f"{system_prompt}\n\n{tools_desc}"

    messages = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": user_message},
    ]

    for _ in range(max_turns):
        # LLM 生成
        try:
            action = llm_fn(messages, temperature)
        except Exception as e:
            logger.error(f"LLM 生成失败: {e}")
            break

        messages.append({"role": "assistant", "content": action})

        # 环境执行（解析 tool_call + 执行 + tracking 全在 env.step 里）
        obs, reward, done, info = env.step(action)

        # 追加 observation
        messages.append({"role": "user", "content": f"<tool_response>\n{obs}\n</tool_response>"})

        if done:
            break

    tracking = env.get_tracking_variables()
    return messages, tracking
