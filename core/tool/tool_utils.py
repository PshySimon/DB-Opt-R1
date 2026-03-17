"""
工具装饰器，将普通函数转为 Tool 对象
"""

import inspect
import json
from typing import Callable, Optional, Dict, Any, get_type_hints

from .tool_base import Tool


def _parse_docstring(func) -> tuple:
    """从 Google-style docstring 提取描述和参数说明"""
    doc = inspect.getdoc(func) or ""
    lines = doc.strip().split("\n")

    description = ""
    params = {}
    section = "desc"

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            section = "args"
            continue
        elif stripped.lower().startswith("returns:"):
            section = "returns"
            continue

        if section == "desc":
            description += stripped + " "
        elif section == "args":
            if stripped.startswith("-") or ":" in stripped:
                # "param_name: description" or "- param_name: description"
                stripped = stripped.lstrip("- ")
                if ":" in stripped:
                    pname, pdesc = stripped.split(":", 1)
                    params[pname.strip()] = pdesc.strip()

    return description.strip(), params


def _python_type_to_json(type_hint) -> str:
    """Python 类型 → JSON Schema 类型"""
    mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    return mapping.get(type_hint, "string")


def function_to_tool(func: Callable, name: str = None,
                     description: str = None) -> Tool:
    """将普通函数转为 Tool 对象"""

    func_name = name or func.__name__
    func_desc, param_docs = _parse_docstring(func)
    func_desc = description or func_desc

    # 构建 JSON Schema 参数
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    properties = {}
    required = []

    for pname, param in sig.parameters.items():
        ptype = hints.get(pname, str)
        pdesc = param_docs.get(pname, "")
        properties[pname] = {
            "type": _python_type_to_json(ptype),
            "description": pdesc,
        }
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    parameters = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    class FunctionTool(Tool):
        def __init__(self):
            super().__init__(
                name=func_name,
                description=func_desc,
                parameters=parameters
            )
            self.func = func

        def execute(self, args: Dict[str, Any]) -> str:
            valid_args = {
                k: v for k, v in args.items()
                if k in sig.parameters
            }
            try:
                result = self.func(**valid_args)
                return result if isinstance(result, str) else str(result)
            except Exception as e:
                return f"Error: {str(e)}"

    return FunctionTool()


def tool_decorator(name: str = None, description: str = None):
    """装饰器：将函数转为 Tool"""
    def decorator(func: Callable) -> Tool:
        return function_to_tool(func, name=name, description=description)
    return decorator
