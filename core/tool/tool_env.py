"""
工具环境，复用 Agent-R1 设计
"""

import re
import json
from typing import Dict, List, Any
from copy import deepcopy

from .tool_base import Tool


INVALID_TOOL_CALL_MESSAGE = (
    "Invalid tool call format. Please use "
    "<tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>"
)


def _tool_fingerprint(tool_name: str, tool_args: Dict[str, Any]) -> str:
    return json.dumps({"tool": tool_name, "args": tool_args}, sort_keys=True, ensure_ascii=False)


def step(env: 'ToolEnv', action_text: str):
    """执行一步工具交互，返回 (observation, reward, done, info)"""
    env.steps_taken += 1
    action = env.extract_tool_call(action_text)

    if action == env.INVALID_ACTION:
        env.tool_execution_error_streak = 0
        env.invalid_tool_call_streak += 1
        done = env.invalid_tool_call_streak >= env.max_invalid_tool_call_streak
        if done:
            env.termination_reason = "invalid_tool_call"
        result = INVALID_TOOL_CALL_MESSAGE
        reward = env.PENALTY_FOR_INVALID
        env._update_tracking(action_text, action, False, False, reward)
        return result, reward, done, {"action_is_valid": False, "action_is_effective": False}

    tool_name = action["tool"]
    tool_args = action["args"]
    env.invalid_tool_call_streak = 0

    if tool_name not in env.tool_map:
        env.tool_execution_error_streak = 0
        result = f"Unknown tool: {tool_name}"
        reward = env.PENALTY_FOR_INEFFECTIVE
        env.invalid_tool_call_streak += 1
        done = env.invalid_tool_call_streak >= env.max_invalid_tool_call_streak
        if done:
            env.termination_reason = "invalid_tool_call"
        env._update_tracking(action_text, action, False, False, reward)
        return result, reward, done, {"action_is_valid": False, "action_is_effective": False}

    tool = env.tool_map[tool_name]

    is_valid, error_msg = tool.validate_args(tool_args)
    if not is_valid:
        env.tool_execution_error_streak = 0
        result = f"Invalid arguments for '{tool_name}': {error_msg}"
        reward = env.PENALTY_FOR_INEFFECTIVE
        env._update_tracking(action_text, action, True, False, reward)
        return result, reward, False, {"action_is_valid": True, "action_is_effective": False}

    fingerprint = _tool_fingerprint(tool_name, tool_args)
    if env.last_tool_fingerprint == fingerprint:
        env.same_tool_same_args_streak += 1
    else:
        env.same_tool_same_args_streak = 1
        env.last_tool_fingerprint = fingerprint

    if tool_name == "predict_performance":
        env.predict_calls_used += 1

    if tool_name == "finish_tuning":
        result = tool.execute(tool_args)
        reward = tool.calculate_reward(tool_args, result)
        env.tool_history.append({"tool": tool_name, "args": tool_args, "result": result})
        env.last_valid_tool_call = {"tool": tool_name, "args": tool_args}
        env.last_valid_config = deepcopy(env.get_current_config_snapshot())
        env.termination_reason = "finish_tuning"
        env._update_tracking(action_text, action, True, True, reward)
        return result, reward, True, {"action_is_valid": True, "action_is_effective": True}

    if tool_name == "predict_performance" and env.predict_calls_used >= env.max_predict_calls:
        predict_budget_done = True
    else:
        predict_budget_done = False

    repeated_tool_done = (
        tool_name != "predict_performance"
        and env.same_tool_same_args_streak >= env.max_same_tool_same_args_streak
    )

    try:
        result = tool.execute(tool_args)
        env.tool_execution_error_streak = 0
        reward = tool.calculate_reward(tool_args, result)
        env.tool_history.append({"tool": tool_name, "args": tool_args, "result": result})
        env.last_valid_tool_call = {"tool": tool_name, "args": tool_args}
        env.last_valid_config = deepcopy(env.get_current_config_snapshot())
        done = False
        if predict_budget_done:
            env.termination_reason = "predict_budget_exhausted"
            done = True
        elif repeated_tool_done:
            env.termination_reason = "repeated_tool_call"
            done = True
        elif env.steps_taken >= env.max_turns:
            env.termination_reason = "max_turns_reached"
            done = True
        env._update_tracking(action_text, action, True, True, reward)
        return result, reward, done, {"action_is_valid": True, "action_is_effective": True}
    except Exception as e:
        result = f"Error executing '{tool_name}': {str(e)}"
        reward = env.PENALTY_FOR_INEFFECTIVE
        env.tool_execution_error_streak += 1
        done = env.tool_execution_error_streak >= env.max_tool_execution_error_streak
        if done:
            env.termination_reason = "tool_execution_error"
        env._update_tracking(action_text, action, True, False, reward)
        return result, reward, done, {"action_is_valid": True, "action_is_effective": False}


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
        self.termination_reason = None
        self.invalid_tool_call_streak = 0
        self.max_invalid_tool_call_streak = 2
        self.tool_execution_error_streak = 0
        self.max_tool_execution_error_streak = 2
        self.predict_calls_used = 0
        self.max_predict_calls = 3
        self.same_tool_same_args_streak = 0
        self.max_same_tool_same_args_streak = 3
        self.last_tool_fingerprint = None
        self.last_valid_tool_call = None
        self.last_valid_config = None

    def step(self, action_text: str):
        """执行一步工具交互，返回 (observation, reward, done, info)"""
        return step(self, action_text)

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
            raw = match.group(1).strip()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data, _ = json.JSONDecoder().raw_decode(raw)
            if not isinstance(data, dict):
                return self.INVALID_ACTION
            tool_name = data.get("name")
            tool_args = data.get("arguments", {})
            if not isinstance(tool_name, str):
                return self.INVALID_ACTION
            if not isinstance(tool_args, dict):
                return self.INVALID_ACTION
            return {"tool": tool_name, "args": tool_args}
        except (json.JSONDecodeError, Exception):
            return self.INVALID_ACTION

    def get_tracking_variables(self) -> Dict:
        return {
            "rewards": self.rewards,
            "total_reward": sum(self.rewards),
            "steps_taken": self.steps_taken,
            "tool_history": self.tool_history,
            "valid_actions": sum(1 for a in self._actions_valid if a is not None),
            "effective_actions": sum(1 for a in self._actions_effective if a is not None),
            "termination_reason": self.termination_reason,
            "predict_calls_used": self.predict_calls_used,
            "last_valid_tool_call": self.last_valid_tool_call,
            "last_valid_config": self.last_valid_config,
        }

    def _update_tracking(self, response, action, valid, effective, reward):
        self._actions.append(response)
        self._actions_valid.append(action if valid else None)
        self._actions_effective.append(action if effective else None)
        self.rewards.append(reward)

    def get_current_config_snapshot(self):
        return None

    def copy(self):
        env = ToolEnv(tools=self.tools, max_turns=self.max_turns)
        env.tool_history = deepcopy(self.tool_history)
        env.rewards = deepcopy(self.rewards)
        env.steps_taken = self.steps_taken
        env._actions = deepcopy(self._actions)
        env._actions_valid = deepcopy(self._actions_valid)
        env._actions_effective = deepcopy(self._actions_effective)
        env.termination_reason = self.termination_reason
        env.invalid_tool_call_streak = self.invalid_tool_call_streak
        env.max_invalid_tool_call_streak = self.max_invalid_tool_call_streak
        env.predict_calls_used = self.predict_calls_used
        env.max_predict_calls = self.max_predict_calls
        env.same_tool_same_args_streak = self.same_tool_same_args_streak
        env.max_same_tool_same_args_streak = self.max_same_tool_same_args_streak
        env.last_tool_fingerprint = self.last_tool_fingerprint
        env.last_valid_tool_call = deepcopy(self.last_valid_tool_call)
        env.last_valid_config = deepcopy(self.last_valid_config)
        return env
