"""
Tool 基类，复用 Agent-R1 设计
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List


class Tool(ABC):
    """工具基类"""

    def __init__(self, name: str, description: str, parameters: Dict = None):
        self.name = name
        self.description = description
        self.parameters = parameters or {
            "type": "object",
            "properties": {},
            "required": []
        }
        if "type" not in self.parameters:
            self.parameters["type"] = "object"
        if "properties" not in self.parameters:
            self.parameters["properties"] = {}
        if "required" not in self.parameters:
            self.parameters["required"] = []

    def get_description(self) -> Dict:
        """返回 OpenAI function calling 格式的工具描述"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def get_simple_description(self) -> str:
        """返回人类可读的工具描述"""
        desc = f"Tool name: {self.name}\nDescription: {self.description}"
        if self.parameters and "properties" in self.parameters:
            properties = self.parameters["properties"]
            required = self.parameters.get("required", [])
            if properties:
                desc += "\nParameters:"
                for name, info in properties.items():
                    param_desc = info.get("description", "")
                    is_required = "(Required)" if name in required else "(Optional)"
                    desc += f"\n  - {name} {is_required}: {param_desc}"
        return desc

    @abstractmethod
    def execute(self, args: Dict) -> str:
        pass

    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        return [self.execute(args) for args in args_list]

    def validate_args(self, args: Dict) -> Tuple[bool, str]:
        if not isinstance(args, dict):
            return False, "Arguments must be a dictionary"
        required_params = self.parameters.get("required", [])
        for param in required_params:
            if param not in args:
                return False, f"Missing required parameter: {param}"
        return True, "OK"

    def calculate_reward(self, args: Dict, result: str) -> float:
        return 0.0
