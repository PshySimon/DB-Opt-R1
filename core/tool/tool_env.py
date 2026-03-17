"""
工具环境，复用 Agent-R1 设计
"""

import re
import json
from typing import Dict, List, Any
from copy import deepcopy

from .tool_base import Tool


def step(env: 'ToolEnv', action_text: str):
    """执行一步工具交互，返回 (observation, reward, done, info)"""
    env.steps_taken += 1
    action = env.extract_tool_call(action_text)

    if action == env.INVALID_ACTION:
        result = "Invalid tool call format. Please use <tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>"
        reward = env.PENALTY_FOR_INVALID
        env._update_tracking(action_text, action, False, False, reward)
        return result, reward, False, {"action_is_valid": False, "action_is_effective": False}

    tool_name = action["tool"]
    tool_args = action["args"]

    if tool_name not in env.tool_map:
        result = f"Unknown tool: {tool_name}"
        reward = env.PENALTY_FOR_INEFFECTIVE
        env._update_tracking(action_text, action, True, False, reward)
        return result, reward, False, {"action_is_valid": True, "action_is_effective": False}

    tool = env.tool_map[tool_name]

    is_valid, error_msg = tool.validate_args(tool_args)
    if not is_valid:
        result = f"Invalid arguments for '{tool_name}': {error_msg}"
        reward = env.PENALTY_FOR_INEFFECTIVE
        env._update_tracking(action_text, action, True, False, reward)
        return result, reward, False, {"action_is_valid": True, "action_is_effective": False}

    try:
        result = tool.execute(tool_args)
        reward = tool.calculate_reward(tool_args, result)
        env.tool_history.append({"tool": tool_name, "args": tool_args, "result": result})
        done = env.steps_taken >= env.max_turns
        env._update_tracking(action_text, action, True, True, reward)
        return result, reward, done, {"action_is_valid": True, "action_is_effective": True}
    except Exception as e:
        result = f"Error executing '{tool_name}': {str(e)}"
        reward = env.PENALTY_FOR_INEFFECTIVE
        env._update_tracking(action_text, action, True, False, reward)
        return result, reward, False, {"action_is_valid": True, "action_is_effective": False}


class ToolEnv:
    """通用工具环境"""

    INVALID_ACTION = {"tool": "invalid", "args": {}}
    PENALTY_FOR_INVALID = -0.1
    PENALTY_FOR_INEFFECTIVE = -0.05

    def __init__(self, tools: List[Tool] = None, max_turns: int = 10):
        self.tools = tools or []
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.max_turns = max_turns
        self.reset()

    def reset(self):
        self.rewards = []
        self.tool_history = []
        self.steps_taken = 0
        self._actions = []
        self._actions_valid = []
        self._actions_effective = []

    def tools_format_func(self) -> str:
        """生成工具描述 prompt"""
        template = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""
        tools = "\n".join([
            json.dumps(tool.get_description(), ensure_ascii=False)
            for tool in self.tools
        ])
        return template.format(tools=tools)

    def extract_tool_call(self, text: str) -> Dict:
        """从 LLM 输出解析 <tool_call>...</tool_call>"""
        match = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
        if not match:
            return self.INVALID_ACTION
        try:
            data = json.loads(match.group(1).strip())
            if "name" not in data:
                return self.INVALID_ACTION
            return {"tool": data["name"], "args": data.get("arguments", {})}
        except (json.JSONDecodeError, Exception):
            return self.INVALID_ACTION

    def get_tracking_variables(self) -> Dict:
        return {
            "rewards": self.rewards,
            "total_reward": sum(self.rewards),
            "steps_taken": self.steps_taken,
            "tool_history": self.tool_history,
        }

    def _update_tracking(self, response, action, valid, effective, reward):
        self._actions.append(response)
        self._actions_valid.append(action if valid else None)
        self._actions_effective.append(action if effective else None)
        self.rewards.append(reward)

    def copy(self):
        env = ToolEnv(tools=self.tools, max_turns=self.max_turns)
        env.tool_history = deepcopy(self.tool_history)
        env.rewards = deepcopy(self.rewards)
        env.steps_taken = self.steps_taken
        env._actions = deepcopy(self._actions)
        env._actions_valid = deepcopy(self._actions_valid)
        env._actions_effective = deepcopy(self._actions_effective)
        return env
