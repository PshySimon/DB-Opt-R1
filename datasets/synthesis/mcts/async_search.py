"""
异步并发 MCTS 搜索

通过 Virtual Loss + ThreadPool 实现单棵树内多个 simulation 并发执行，
保持 LLM API 始终有多个请求在飞，大幅减少搜索耗时。
"""

import json
import logging
import threading
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from .node import MCTSNode
from .search import MCTSSearch

logger = logging.getLogger(__name__)


class AsyncMCTSSearch(MCTSSearch):
    """并发版 MCTS：多个 simulation 异步并行"""

    def __init__(self, env_factory: Callable, llm_generate: Callable,
                 config: dict = None):
        """
        Args:
            env_factory: 无参函数，每次调用返回一个新的 env 实例
                         （每个线程需要独立的 env 避免状态冲突）
            llm_generate: LLM 生成函数 (prompt, temperature) -> str
            config: 搜索参数 + num_workers (并发数)
        """
        self.env_factory = env_factory
        cfg = config or {}
        self.num_workers = cfg.get("num_workers", 4)

        # 用一个临时 env 初始化父类（获取 tools 描述等）
        temp_env = env_factory()
        super().__init__(env=temp_env, llm_generate=llm_generate, config=config)

    def search(self, sample_idx: int = None) -> MCTSNode:
        """并发执行 MCTS 搜索"""
        root = MCTSNode()
        root.sample_idx = sample_idx
        lock = threading.Lock()

        completed = 0
        total = self.num_simulations

        logger.info(f"  AsyncMCTS: {total} simulations, {self.num_workers} workers")

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {}

            # 提交初始批次
            initial_batch = min(self.num_workers, total)
            for _ in range(initial_batch):
                with lock:
                    node = self._select_with_virtual_loss(root)
                future = pool.submit(
                    self._simulation_job, node, root, sample_idx, lock
                )
                futures[future] = node

            # 完成一个就提交下一个
            while completed < total:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    node = futures.pop(future)
                    try:
                        child, reward = future.result()
                        with lock:
                            self._remove_virtual_loss(node)
                            self._real_backpropagate(child, reward)
                    except Exception as e:
                        logger.error(f"  simulation 失败: {e}")
                        with lock:
                            self._remove_virtual_loss(node)

                    completed += 1

                    if completed % 10 == 0:
                        best_info = ""
                        if root.children:
                            best = root.best_child_by_reward()
                            best_info = f", best_reward={best.avg_reward:.3f}"
                        logger.info(
                            f"  simulation {completed}/{total}{best_info}"
                        )

                    # 还有余量就提交下一个
                    if completed + len(futures) < total:
                        with lock:
                            next_node = self._select_with_virtual_loss(root)
                        f = pool.submit(
                            self._simulation_job, next_node, root,
                            sample_idx, lock
                        )
                        futures[f] = next_node

        return root

    def _simulation_job(self, node: MCTSNode, root: MCTSNode,
                        sample_idx: int, lock: threading.Lock):
        """
        单个线程的完整任务：expand → simulate → 返回 (child, reward)

        每个线程用自己的 env 实例，避免状态冲突。
        """
        # 创建线程本地 env
        env = self.env_factory()

        # Expand（需要 env 执行 action 获取 observation）
        with lock:
            trajectory = list(node.trajectory)
            needs_expand = node.depth < self.max_depth and \
                           not node.is_fully_expanded(self.max_children)

        if needs_expand:
            child = self._expand_with_env(node, env, sample_idx, lock)
        else:
            child = node

        # Simulate
        reward = self._simulate_with_env(child, env, sample_idx, lock)
        return child, reward

    def _expand_with_env(self, node: MCTSNode, env, sample_idx: int,
                         lock: threading.Lock) -> MCTSNode:
        """使用线程本地 env 执行 expand"""
        with lock:
            prompt = self._build_prompt(node.trajectory)

        # 生成候选（LLM 调用不需要锁）
        action = self.llm_generate(prompt, temperature=self.expand_temperature)
        action = action.strip()

        if not action:
            return node

        # 用线程本地 env 执行 action 获取 observation
        env.reset(sample_idx=sample_idx)
        with lock:
            trajectory = node.trajectory
        for step in trajectory:
            env.step(step["action"])
        env.steps_taken = 0

        obs, reward, done, info = env.step(action)

        # 加锁修改树
        with lock:
            # 检查是否已存在相同 action 的子节点
            existing = [c for c in node.children if c.action == action]
            if existing:
                return existing[0]
            child = node.add_child(action=action, observation=obs)
            return child

    def _simulate_with_env(self, node: MCTSNode, env, sample_idx: int,
                           lock: threading.Lock) -> float:
        """使用线程本地 env 执行 simulate"""
        # 重置并回放
        env.reset(sample_idx=sample_idx)
        with lock:
            trajectory = list(node.trajectory)

        for step in trajectory:
            env.step(step["action"])
        env.steps_taken = 0

        # Rollout
        rollout_steps = []
        remaining = self.max_depth - node.depth
        for _ in range(remaining):
            prompt = self._build_prompt(trajectory)
            action = self.llm_generate(prompt, temperature=self.rollout_temperature)
            action = action.strip()

            obs, reward, done, info = env.step(action)
            step = {"action": action, "observation": obs}
            trajectory.append(step)
            rollout_steps.append(step)

            # 检查 tool_history 而不是手动解析 tool_call
            if env.tool_history and env.tool_history[-1].get("tool") == "predict_performance":
                break

        # 通过 env.step 调 predict_performance 计算 reward
        obs, _, _, _ = env.step(
            '<tool_call>\n{"name": "predict_performance", "arguments": {}}\n</tool_call>'
        )
        try:
            parsed = json.loads(obs)
            reward = parsed.get("improvement_pct", 0) / 100.0
        except (json.JSONDecodeError, TypeError):
            reward = 0.0

        # 保存最优 rollout
        with lock:
            if not node.rollout_trajectory or reward > node.avg_reward:
                node.rollout_trajectory = rollout_steps

        return reward

    # ===== Virtual Loss =====

    def _select_with_virtual_loss(self, root: MCTSNode) -> MCTSNode:
        """
        UCB1 选择 + Virtual Loss

        选中路径上每个节点 visit_count += 1（降低其 UCB1 分数），
        让其他并发线程自然避开同一路径。
        """
        node = root
        node.visit_count += 1  # root 的 virtual loss

        while not node.is_leaf() and node.is_fully_expanded(self.max_children):
            node = node.best_child(self.ucb_c)
            node.visit_count += 1  # virtual loss

        return node

    def _remove_virtual_loss(self, node: MCTSNode):
        """撤销 virtual loss：沿路径到 root，visit_count -= 1"""
        current = node
        while current is not None:
            current.visit_count -= 1
            current = current.parent

    def _real_backpropagate(self, node: MCTSNode, reward: float):
        """真正的 backprop（不含 virtual loss 的那 +1）"""
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent


