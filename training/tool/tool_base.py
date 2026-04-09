"""
Training compatibility layer for tools.

The single source of truth for tool semantics lives in core.tool.tool_base.
"""

from core.tool.tool_base import Tool

__all__ = ["Tool"]
