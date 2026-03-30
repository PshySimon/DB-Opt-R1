import os
import sys
import json
import time
import logging
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
        self.lock = threading.Lock()
        
        # 封禁与冷却
        self.banned = False
        self.cooldown_until = 0.0

    def try_acquire(self) -> bool:
        with self.lock:
            if self.active_requests < self.max_concurrent:
                self.active_requests += 1
                return True
            return False

    def release(self):
        with self.lock:
            self.active_requests -= 1

class MultiProviderLLMClient:
    """多中转站 LLM 客户端，采用严格轮询，依靠独立槽位自然实现速度加权，支持硬错误封禁与软错误冷却"""
    
    def __init__(self, target_model: str, providers_config: str = None, 
                 single_api_key: str = None, single_api_base: str = None):
        self.target_model = target_model
        self.stats: List[ClientStats] = []
        self._rr_index = 0
        self._rr_lock = threading.Lock()
        
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
        """严格轮询拿一个当前可用（未封禁、未冷却、且并发槽位未满）的节点"""
        with self._rr_lock:
            start_idx = self._rr_index
            for i in range(self.total_clients):
                idx = (start_idx + i) % self.total_clients
                c = self.stats[idx]
                
                # 检查不可用状态
                if c.banned:
                    continue
                if time.time() < c.cooldown_until:
                    continue
                    
                if c.try_acquire():
                    # 轮询指针拨到下一个
                    self._rr_index = (idx + 1) % self.total_clients
                    return c
                    
            # 全部处于被 ban、冷却中或者并发打满状态
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
                # 空闲轮空，退避等待调度（不计入 API 错误次数）
                time.sleep(0.5)
                continue
                
            try:
                response = selected.client.chat.completions.create(
                    model=selected.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=2048,
                )
                res_content = response.choices[0].message.content
                if not res_content:
                    raise Exception("返回内容为空 (completion_tokens: 0)")
                    
                selected.release()
                return res_content
                
            except Exception as e:
                selected.release()
                last_error = e
                err_str = str(e).lower()
                
                # 致命错误关键词（如模型不支持、余额不足、无权限）
                fatal_keywords = ["model_not_found", "unsupported", "does not exist", 
                                  "401", "404", "insufficient_quota", "invalid_api_key"]
                
                if any(k in err_str for k in fatal_keywords):
                    selected.banned = True
                    logger.error(f"❌ 致命错误警告: 节点 {selected.api_base} 触发不可恢复错误，已被永久禁用! [{e}]")
                else:
                    # 一般的并发过高 429、网络超时、502、503 等，赋予 30 秒冷却期
                    selected.cooldown_until = time.time() + 30
                    logger.warning(f"⚠️ 临时错误警告: 节点 {selected.api_base} 并发/网络异常，冷却 30 秒... [{e}]")
                
                error_count += 1
                if error_count >= max_error_retries:
                    break
                time.sleep(1)
        
        logger.error(f"所有 API 提供商暂时均调用失败或一直繁忙！共重试 {error_count} 次。最后报错: {last_error}")
        raise Exception(f"All API providers failed after {error_count} retries. Last error: {last_error}")
