"""
Training compatibility layer for tool environments.

Single-step ToolEnv semantics come from core.tool.tool_env.
The only training-specific helper kept here is step_batch.
"""

from typing import List

from core.tool.tool_env import ToolEnv as CoreToolEnv
from core.tool.tool_env import step as core_step


class ToolEnv(CoreToolEnv):
    """Compatibility wrapper that preserves the training-side `tool_desc` field."""

    def __init__(self, tools=None, max_turns: int = 10):
        super().__init__(tools=tools, max_turns=max_turns)
        self.tool_desc = [tool.get_description() for tool in self.tools]


def step(env: ToolEnv, action_text: str):
    """Reuse the core single-step semantics exactly."""
    return core_step(env, action_text)


def step_batch(envs: List[ToolEnv], action_texts: List[str]):
    """
    Execute a batch by repeatedly applying the shared single-step semantics.

    This preserves train/inference consistency. If batch optimization is needed
    later, it must prove equivalence to repeated `core_step`.
    """
    assert len(envs) == len(action_texts), "Number of environments and actions must match"
    return [core_step(env, action_text) for env, action_text in zip(envs, action_texts)]


__all__ = ["ToolEnv", "step", "step_batch"]
