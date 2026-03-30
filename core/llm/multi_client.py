import os
import sys
import json
import time
import logging
import random
import threading
from typing import List

logger = logging.getLogger(__name__)

class ClientStats:
    """包装单个 OpenAI client，维护其并发状态与封禁/冷却状态"""
    def __init__(self, idx: int, api_base: str, client, max_concurrent: int, model_name: str):
        self.idx = idx
        self.api_base = api_base
        self.client = client
        self.max_concurrent = max_concurrent
        self.model_name = model_name
        
        self.active_requests = 0
        self.successes = 0
        self.total_latency = 0.0
        
        self.lock = threading.Lock()
        
        # 封禁与冷却
        self.banned = False
        self.cooldown_until = 0.0
        self.error_streak = 0  # 连续报错次数，用于指数退避

    @property
    def avg_latency(self) -> float:
        if self.successes == 0:
            return 2.0  # 未测试过的节点，默认假定 2 秒的延迟
        return self.total_latency / self.successes

    def try_acquire(self) -> bool:
        with self.lock:
            if self.active_requests < self.max_concurrent:
                self.active_requests += 1
                return True
            return False

    def release(self, success: bool, latency: float = 0.0):
        with self.lock:
            self.active_requests -= 1
            if success:
                self.successes += 1
                self.error_streak = 0
                self.total_latency += latency

class MultiProviderLLMClient:
    """多中转站 LLM 客户端，采用基于响应速度的加权随机轮询，并支持指数退避冷却"""
    
    def __init__(self, target_model: str, providers_config: str = None, 
                 single_api_key: str = None, single_api_base: str = None):
        self.target_model = target_model
        self.stats: List[ClientStats] = []
        
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("需要安装 openai: pip install openai")
            sys.exit(1)
            
        default_headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        
        providers = []
        if providers_config and os.path.isfile(providers_config):
            try:
                with open(providers_config, "r", encoding="utf-8") as f:
                    all_providers = json.load(f)
                for p in all_providers:
                    providers.append(p)
                if not providers:
                    logger.error(f"配置文件 {providers_config} 为空！")
                    sys.exit(1)
                logger.info(f"配置由 providers.json 接管，成功加载 {len(providers)} 个 API 提供商配置。")
            except Exception as e:
                logger.error(f"读取 providers 配置文件失败: {e}")
                sys.exit(1)
        elif single_api_key:
            providers = [{
                "api_key": single_api_key,
                "api_base": single_api_base,
                "model": target_model,
                "max_concurrent": 5
            }]
        else:
            logger.error("必须提供 --providers-config 或 --api-key！")
            sys.exit(1)
            
        for i, config in enumerate(providers):
            client = OpenAI(
                api_key=config.get("api_key"),
                base_url=config.get("api_base"),
                default_headers=default_headers
            )
            max_c = config.get("max_concurrent", 5)
            model_n = config.get("model", target_model)
            self.stats.append(ClientStats(i, config.get("api_base"), client, max_c, model_n))
            
        self.total_clients = len(self.stats)

    def _get_next_client(self) -> ClientStats:
        """基于响应速度进行加权轮询挑选一个可用节点（越快越容易被选中）"""
        available_clients = []
        weights = []
        
        for c in self.stats:
            if c.banned or time.time() < c.cooldown_until:
                continue
            # 在没有锁的情况下先大致判断一下并发，如果满了就直接跳过
            if c.active_requests >= c.max_concurrent:
                continue
                
            available_clients.append(c)
            # 权重与延迟成反比，越快权重越大
            weights.append(1.0 / max(c.avg_latency, 0.1))
            
        if not available_clients:
            return None
            
        # 根据速度计算的权重进行随机掷骰子（Roulette Wheel Selection）
        candidates = random.choices(available_clients, weights=weights, k=len(available_clients))
        
        # 依次尝试获取锁，防止被其它线程刚好抢走
        for c in candidates:
            if c.try_acquire():
                return c
                
        return None

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        last_error = None
        # 当因为 API 真报错导致的退避次数上限（防止全网大罢工导致的无限假死）
        max_error_retries = self.total_clients * 3
        error_count = 0
        
        while True:
            # 存活检查
            active_clients = [c for c in self.stats if not c.banned]
            if not active_clients:
                raise Exception(f"所有 API 提供商均触发致命错误被永久封禁！最后报错: {last_error}")
                
            selected = self._get_next_client()
            
            if not selected:
                # 空闲轮空，毫无可用槽位，退避等待调度（不计入 API 错误次数）
                time.sleep(0.5)
                continue
                
            try:
                t0 = time.time()
                response = selected.client.chat.completions.create(
                    model=selected.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=2048,
                )
                res_content = response.choices[0].message.content
                elapsed = time.time() - t0
                
                if not res_content:
                    raise Exception("返回内容为空 (completion_tokens: 0)")
                    
                selected.release(success=True, latency=elapsed)
                return res_content
                
            except Exception as e:
                selected.release(success=False)
                last_error = e
                err_str = str(e).lower()
                
                # 致命错误关键词
                fatal_keywords = ["model_not_found", "unsupported", "does not exist", 
                                  "401", "404", "insufficient_quota", "invalid_api_key"]
                
                if any(k in err_str for k in fatal_keywords):
                    selected.banned = True
                    logger.error(f"❌ 致命错误警告: 节点 {selected.api_base} 触发不可恢复错误，已被永久禁用! [{e}]")
                else:
                    # 指数退避冷却
                    selected.error_streak += 1
                    # 基础 30 秒，每次连续报错翻倍 (30, 60, 120...)，封顶 600 秒
                    cooldown_secs = min(30 * (2 ** (selected.error_streak - 1)), 600)
                    selected.cooldown_until = time.time() + cooldown_secs
                    logger.warning(
                        f"⚠️ 临时错误警告: 节点 {selected.api_base} 网络异常或过载（连续崩溃 {selected.error_streak} 次）"
                        f"，进入指数退避，冷却 {cooldown_secs} 秒... [{e}]"
                    )
                
                error_count += 1
                if error_count >= max_error_retries:
                    break
                time.sleep(1)
        
        logger.error(f"所有 API 提供商暂时均调用失败！共重试 {error_count} 次。最后报错: {last_error}")
        raise Exception(f"All API providers failed after {error_count} retries. Last error: {last_error}")
