import os
import sys
import json
import time
import logging
import itertools
import threading
from typing import List, Dict

logger = logging.getLogger(__name__)


class MultiProviderLLMClient:
    """多中转站 LLM 客户端，支持轮询负载均衡和报错自动顺延"""
    
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
        self.clients = []
        self._lock = threading.Lock()
        
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
                "model": target_model
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
            # 存下引用和它的索引以便打印报错归属
            self.clients.append((i, config.get("api_base"), client))
            
        self.client_cycle = itertools.cycle(self.clients)
        self.total_clients = len(self.clients)
        
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """对外暴露的统一 generate 接口，带有轮询与重试"""
        
        max_retries = self.total_clients
        last_error = None
        
        for attempt in range(max_retries):
            with self._lock:
                idx, api_base, client = next(self.client_cycle)
                
            try:
                response = client.chat.completions.create(
                    model=self.target_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=2048,
                )
                res_content = response.choices[0].message.content
                if not res_content:
                    raise Exception("返回内容为空 (completion_tokens: 0)")
                return res_content
                
            except Exception as e:
                last_error = e
                logger.warning(f"中转站异常 [idx={idx}, base={api_base}]: {e}。将自动切换下一个 API 重试...")
                time.sleep(1)
        
        logger.error(f"所有 API 提供商均调用失败！最后报错: {last_error}")
        raise Exception(f"All API providers failed. Last error: {last_error}")
