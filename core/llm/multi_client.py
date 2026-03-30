import os
import sys
import json
import time
import logging
import threading
from typing import List, Dict

logger = logging.getLogger(__name__)

class ClientStats:
    """包装单个 OpenAI client，维护其并发状态与成功率"""
    def __init__(self, idx: int, api_base: str, client, max_concurrent: int = 5):
        self.idx = idx
        self.api_base = api_base
        self.client = client
        self.max_concurrent = max_concurrent
        
        self.successes = 0
        self.failures = 0
        self.active_requests = 0
        self.lock = threading.Lock()
        
    @property
    def score(self) -> float:
        """使用拉普拉斯平滑计算优先级分数，避免 0 成功率后永远不被调度"""
        return (self.successes + 1.0) / (self.successes + self.failures + 2.0)

    def try_acquire(self) -> bool:
        with self.lock:
            if self.active_requests < self.max_concurrent:
                self.active_requests += 1
                return True
            return False

    def release(self, success: bool):
        with self.lock:
            self.active_requests -= 1
            if success:
                self.successes += 1
            else:
                self.failures += 1

class MultiProviderLLMClient:
    """多中转站 LLM 客户端，支持按成功率优先级排序和最大并发控制"""
    
    def __init__(self, target_model: str, providers_config: str = None, 
                 single_api_key: str = None, single_api_base: str = None):
        """
        Args:
            target_model: 目标使用的模型名（如 gpt-5）
            providers_config: 多中转站配置文件路径（JSON 数组）
            single_api_key: fallback 的单个 api_key
            single_api_base: fallback 的单个 api_base
        """
        self.target_model = target_model
        self.stats: List[ClientStats] = []
        
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("需要安装 openai: pip install openai")
            sys.exit(1)
            
        default_headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}
        
        providers = []
        # 1. 尝试通过配置加载
        if providers_config and os.path.isfile(providers_config):
            try:
                with open(providers_config, "r", encoding="utf-8") as f:
                    all_providers = json.load(f)
                
                # 根据命令行传入的 model 进行过滤
                for p in all_providers:
                    if p.get("model") == target_model:
                        providers.append(p)
                
                if not providers:
                    logger.error(f"配置文件 {providers_config} 中没有找到 model={target_model} 的任何可用供应源！")
                    sys.exit(1)
                
                logger.info(f"成功加载 {len(providers)} 个支持 {target_model} 的 API 提供商配置。")
            except Exception as e:
                logger.error(f"读取 providers 配置文件失败: {e}")
                sys.exit(1)
        # 2. 如果没有配置或解析为空，回退到老逻辑使用单节点
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
            
        # 实例化多个 OpenAI Client 以及包装器
        for i, config in enumerate(providers):
            client = OpenAI(
                api_key=config.get("api_key"),
                base_url=config.get("api_base"),
                default_headers=default_headers
            )
            max_c = config.get("max_concurrent", 5)
            self.stats.append(ClientStats(i, config.get("api_base"), client, max_c))
            
        self.total_clients = len(self.stats)
        
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """对外暴露的统一 generate 接口，基于成功率与并发上限进行智能路由"""
        last_error = None
        
        # 允许重试次数 = 提供商数量 * 3（应对短暂的所有 API 繁忙）
        max_retries = self.total_clients * 3
        
        for attempt in range(max_retries):
            # 1. 按照成功率降序排序；如果成功率相同，优先选择当前并发任务少的
            sorted_clients = sorted(self.stats, key=lambda c: (c.score, -c.active_requests), reverse=True)
            
            # 2. 依次尝试获取并发槽位
            selected = None
            for c in sorted_clients:
                if c.try_acquire():
                    selected = c
                    break
                    
            if not selected:
                # 所有提供商的 5 个并发槽位均满，退避一下再试
                time.sleep(0.5)
                continue
                
            # 3. 拿到槽位，执行请求
            try:
                response = selected.client.chat.completions.create(
                    model=self.target_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=2048,
                )
                res_content = response.choices[0].message.content
                if not res_content:
                    raise Exception("返回内容为空 (completion_tokens: 0)")
                # 成功
                selected.release(success=True)
                return res_content
                
            except Exception as e:
                # 失败
                selected.release(success=False)
                last_error = e
                logger.warning(
                    f"中转站异常 [base={selected.api_base}, score={selected.score:.2f}]: {e}。自动重试..."
                )
                time.sleep(1)
        
        logger.error(f"所有 API 提供商均调用失败或一直繁忙！最后报错: {last_error}")
        raise Exception(f"All API providers failed or busy. Last error: {last_error}")
