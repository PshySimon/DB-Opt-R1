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
    def __init__(self, idx: int, api_base: str, client, max_concurrent: int, model_name: str, api_key_str: str):
        self.idx = idx
        self.api_base = api_base
        self.client = client
        self.api_key_str = api_key_str  # 用于比对重载
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
    """多中转站 LLM 客户端，采用基于响应速度的加权随机轮询，支持指数退避冷却，并每 60 秒热更新配置文件"""
    
    def __init__(self, target_model: str, providers_config: str = None, 
                 single_api_key: str = None, single_api_base: str = None,
                 api_max_concurrent: int = 5):
        self.target_model = target_model
        self.providers_config = providers_config
        self.single_api_key = single_api_key
        self.single_api_base = single_api_base
        self.api_max_concurrent = api_max_concurrent
        
        self.stats: List[ClientStats] = []
        self._stats_lock = threading.RLock()
        
        self.last_reload_time = 0.0
        self.reload_interval = 60.0
        
        self._load_providers(is_reload=False)
        self._health_check()

    def _health_check(self):
        """启动前并发测活所有节点，不可用的直接 ban 掉"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not self.stats:
            return

        logger.info("🏥 启动前测活 {} 个节点...".format(len(self.stats)))

        def _probe(cs: ClientStats) -> bool:
            try:
                from openai import OpenAI
                probe_client = OpenAI(
                    api_key=cs.api_key_str,
                    base_url=cs.api_base,
                    max_retries=0,
                    timeout=10,
                )
                resp = probe_client.chat.completions.create(
                    model=cs.model_name,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=5,
                )
                content = (resp.choices[0].message.content or "").strip()
                return bool(content)
            except Exception:
                return False

        with ThreadPoolExecutor(max_workers=min(len(self.stats), 20)) as pool:
            futures = {pool.submit(_probe, cs): cs for cs in self.stats}
            for fut in as_completed(futures):
                cs = futures[fut]
                alive = fut.result()
                if alive:
                    logger.info("  ✅ {} 可用".format(cs.api_base))
                else:
                    cs.banned = True
                    logger.warning("  ❌ {} 不可用，已禁用".format(cs.api_base))

        alive_count = sum(1 for c in self.stats if not c.banned)
        logger.info("🏥 测活完成: {}/{} 个节点可用".format(alive_count, len(self.stats)))

        if alive_count == 0:
            logger.error("所有节点均不可用！")
            raise RuntimeError("No available API providers after health check")


    def _load_providers(self, is_reload=False):
        try:
            from openai import OpenAI
        except ImportError:
            if not is_reload:
                logger.error("需要安装 openai: pip install openai")
                sys.exit(1)
            return
            
        default_headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        
        providers = []
        if self.providers_config and os.path.isfile(self.providers_config):
            try:
                with open(self.providers_config, "r", encoding="utf-8") as f:
                    all_providers = json.load(f)
                for p in all_providers:
                    providers.append(p)
                if not providers and not is_reload:
                    logger.error(f"配置文件 {self.providers_config} 为空！")
                    sys.exit(1)
            except Exception as e:
                if not is_reload:
                    logger.error(f"读取 providers 配置文件失败: {e}")
                    sys.exit(1)
                else:
                    logger.warning(f"⚠️ 热重载 providers.json 失败 (正在被修改或语法错误?): {e}，按原配置继续运行。")
                    return
        elif self.single_api_key:
            providers = [{
                "api_key": self.single_api_key,
                "api_base": self.single_api_base,
                "model": self.target_model,
                "max_concurrent": self.api_max_concurrent
            }]
        else:
            if not is_reload:
                logger.error("必须提供 --providers-config 或 --api-key！")
                sys.exit(1)
            return

        with self._stats_lock:
            new_stats = []
            changed = False
            
            for i, config in enumerate(providers):
                base = config.get("api_base")
                key = config.get("api_key")
                max_c = config.get("max_concurrent", 5)
                model_n = config.get("model", self.target_model)
                
                existing = None
                for old_c in self.stats:
                    if old_c.api_base == base and old_c.api_key_str == key:
                        existing = old_c
                        break
                        
                if existing:
                    # 如果参数修改了则更新这二项
                    if existing.max_concurrent != max_c or existing.model_name != model_n:
                        existing.max_concurrent = max_c
                        existing.model_name = model_n
                        changed = True
                    new_stats.append(existing)
                else:
                    client = OpenAI(
                        api_key=key,
                        base_url=base,
                        default_headers=default_headers
                    )
                    new_stats.append(ClientStats(i, base, client, max_c, model_n, key))
                    changed = True
                    
            if len(new_stats) != len(self.stats):
                changed = True
                
            if changed or not is_reload:
                self.stats = new_stats
                self.total_clients = len(self.stats)
                if is_reload:
                    logger.info(f"🔄 热重载检测到提供商配置变更，当前同步为 {self.total_clients} 个节点（保留原节点健康分状态）。")
                else:
                    logger.info(f"配置由 JSON 接管，成功加载 {self.total_clients} 个 API 提供商配置。")
                    
            self.last_reload_time = time.time()

    def _get_next_client(self) -> ClientStats:
        """基于响应速度进行加权轮询挑选一个可用节点，并附带 60s 配置文件热更检测"""
        if self.providers_config and time.time() - self.last_reload_time >= self.reload_interval:
            self._load_providers(is_reload=True)
            
        available_clients = []
        weights = []
        
        with self._stats_lock:
            for c in self.stats:
                if c.banned or time.time() < c.cooldown_until:
                    continue
                # 粗略判断并发
                if c.active_requests >= c.max_concurrent:
                    continue
                    
                available_clients.append(c)
                weights.append(1.0 / max(c.avg_latency, 0.1))
                
        if not available_clients:
            return None
            
        # 根据速度权重加权随机挑选（Roulette Wheel Selection）
        candidates = random.choices(available_clients, weights=weights, k=len(available_clients))
        
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
                    stop=["</tool_call>"]
                )
                res_content = response.choices[0].message.content
                
                # 由于设置了 stop 参数，API 返回结果会正好截断在匹配部位且不包含它，这里我们需要手补上去，确保 parser 能够提亮闭合标签
                if res_content and "<tool_call>" in res_content and "</tool_call>" not in res_content:
                    res_content += "</tool_call>"
                    
                elapsed = time.time() - t0
                
                if not res_content:
                    raise Exception("返回内容为空 (completion_tokens: 0)")
                    
                selected.release(success=True, latency=elapsed)
                return res_content
                
            except Exception as e:
                selected.release(success=False)
                last_error = e
                err_str = str(e).lower()
                
                # 致命错误关键词（欠费、不存、封禁）
                fatal_keywords = ["model_not_found", "unsupported", "does not exist", 
                                  "401", "404", "403", "insufficient", "invalid_api_key", "额度不足"]
                
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
