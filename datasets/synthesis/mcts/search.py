"""
MCTS 搜索算法
"""

import json
import random
import logging
from typing import Callable

from .node import MCTSNode

logger = logging.getLogger(__name__)


class MCTSSearch:
    """蒙特卡洛树搜索"""

    def __init__(self, env, llm_generate: Callable, config: dict = None):
        """
        Args:
            env: DBToolEnv 实例（train 模式）
            llm_generate: LLM 生成函数 (prompt, temperature) -> str
            config: 搜索参数
        """
        self.env = env
        self.llm_generate = llm_generate

        cfg = config or {}
        self.num_simulations = cfg.get("num_simulations", 5)
        self.max_children = cfg.get("max_children", 3)
        self.max_depth = cfg.get("max_depth", 10)
        self.ucb_c = cfg.get("ucb_c", 1.414)
        self.expand_temperature = cfg.get("expand_temperature", 0.8)
        self.rollout_temperature = cfg.get("rollout_temperature", 0.3)
        self.system_prompt = cfg.get("system_prompt",
            "你是 PostgreSQL 调优专家。通过调用工具来观察数据库状态、调整参数、验证性能。"
            "目标是最大化 TPS（每秒事务数）。")

    def search(self, sample_idx: int = None) -> MCTSNode:
        """对一个环境样本执行 MCTS 搜索，返回根节点"""
        root = MCTSNode()
        root.sample_idx = sample_idx

        for i in range(self.num_simulations):
            # 1. Selection
            node = self._select(root)

            # 2. Expansion
            if node.depth < self.max_depth:
                child = self._expand(node)
            else:
                child = node

            # 3. Simulation
            reward = self._simulate(child)

            # 4. Backpropagation
            self._backpropagate(child, reward)

            if (i + 1) % 10 == 0:
                logger.info(f"  simulation {i+1}/{self.num_simulations}, "
                           f"best_reward={root.best_child_by_reward().avg_reward:.3f}"
                           if root.children else "")

        return root

    def _select(self, node: MCTSNode) -> MCTSNode:
        """沿 UCB1 最高路径选到叶节点或未完全展开的节点"""
        while not node.is_leaf() and node.is_fully_expanded(self.max_children):
            node = node.best_child(self.ucb_c)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """生成候选行动，创建子节点"""
        prompt = self._build_prompt(node.trajectory)

        # 生成候选行动
        candidates = set()
        attempts = 0
        while len(candidates) < self.max_children and attempts < self.max_children * 2:
            action = self.llm_generate(prompt, temperature=self.expand_temperature)
            action = action.strip()
            if action and action not in candidates:
                candidates.add(action)
            attempts += 1

        # 创建子节点
        new_children = []
        for action in candidates:
            if not any(c.action == action for c in node.children):
                # 在环境中执行获取 observation
                obs = self._execute_action(node, action)
                child = node.add_child(action=action, observation=obs)
                new_children.append(child)

        # 返回一个未访问的子节点
        unvisited = [c for c in node.children if c.visit_count == 0]
        if unvisited:
            return random.choice(unvisited)
        return random.choice(node.children) if node.children else node

    def _simulate(self, node: MCTSNode) -> float:
        """从当前节点 rollout 到结束，返回 reward"""
        # 重置环境到该样本
        root = node
        while root.parent is not None:
            root = root.parent
        self.env.reset(sample_idx=root.sample_idx)

        # 回放已有轨迹
        trajectory = list(node.trajectory)  # 复制，避免污染节点属性
        for step in trajectory:
            self._replay_step(step["action"])

        # 继续 rollout
        rollout_steps = []
        remaining = self.max_depth - node.depth
        for _ in range(remaining):
            prompt = self._build_prompt(trajectory)
            action = self.llm_generate(prompt, temperature=self.rollout_temperature)
            action = action.strip()

            obs = self._execute_in_env(action)
            step = {"action": action, "observation": obs}
            trajectory.append(step)
            rollout_steps.append(step)

            # 如果调了 predict_performance 就结束
            tool_call = self._parse_tool_call(action)
            if tool_call and tool_call.get("name") == "predict_performance":
                break

        reward = self._compute_reward()

        # 保留最优 rollout（一个叶节点可能被多次 simulate）
        if not node.rollout_trajectory or reward > node.avg_reward:
            node.rollout_trajectory = rollout_steps

        return reward

    def _backpropagate(self, node: MCTSNode, reward: float):
        """回传 reward"""
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent

    def _build_prompt(self, trajectory: list) -> str:
        """构建 LLM prompt"""
        tools_desc = self.env.tools_format_func() if hasattr(self.env, 'tools_format_func') else ""
        messages = [f"System: {self.system_prompt}\n\n{tools_desc}\n"]

        # few-shot 示例
        if hasattr(self, '_example_trajectory') and self._example_trajectory:
            messages.append(f"{self._example_trajectory}\n\n---\n现在请你完成以下任务：\n")

        messages.append("User: 请优化这个数据库的性能。\n")

        for step in trajectory:
            messages.append(f"Assistant: {step['action']}\n")
            if step.get("observation"):
                messages.append(f"Observation: {step['observation']}\n")

        messages.append("Assistant: ")
        return "\n".join(messages)

    def _execute_action(self, parent_node: MCTSNode, action: str) -> str:
        """在环境中执行单个行动，返回 observation"""
        root = parent_node
        while root.parent is not None:
            root = root.parent
        self.env.reset(sample_idx=root.sample_idx)

        # 回放到 parent
        for step in parent_node.trajectory:
            self._replay_step(step["action"])

        # 执行新行动
        return self._execute_in_env(action)

    def _replay_step(self, action: str):
        """回放一步"""
        self._execute_in_env(action)

    def _execute_in_env(self, action: str) -> str:
        """在环境中执行 action（tool_call），返回 observation"""
        try:
            # 解析 tool_call
            tool_call = self._parse_tool_call(action)
            if tool_call is None:
                return "Error: invalid tool call format"

            name = tool_call["name"]
            args = tool_call.get("arguments", {})

            # 在工具中执行
            for tool in self.env.tools:
                if tool.name == name:
                    return tool.execute(args)

            return f"Error: tool '{name}' not found"
        except Exception as e:
            return f"Error: {str(e)}"

    def _parse_tool_call(self, action: str) -> dict:
        """从 action 字符串中解析 tool_call"""
        # 支持 <tool_call>...</tool_call> 格式
        import re
        match = re.search(r'<tool_call>(.*?)</tool_call>', action, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None

        # 也支持裸 JSON
        try:
            parsed = json.loads(action)
            if "name" in parsed:
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        return None

    def _compute_reward(self) -> float:
        """计算当前环境状态的 reward"""
        env_state = self.env.env_state
        baseline_tps = env_state.get("tps", 0)
        if baseline_tps <= 0:
            return 0.0

        # 用 predict_performance 工具获取预测
        for tool in self.env.tools:
            if tool.name == "predict_performance":
                result = tool.execute({})
                try:
                    parsed = json.loads(result)
                    pred_tps = parsed.get("predicted_tps", 0)
                    return (pred_tps - baseline_tps) / baseline_tps
                except (json.JSONDecodeError, TypeError):
                    return 0.0

        return 0.0
