"""
db-opt-r1 统一配置管理
"""

import yaml
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Config:
    """统一配置类，支持 dot notation 访问"""

    def __init__(self, config_path: str):
        self._path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)
        logger.info(f"加载配置: {config_path}")

    def get(self, key: str, default: Any = None) -> Any:
        """支持 dot notation 读取配置

        Examples:
            config.get("general.seed")           → 42
            config.get("tools.database.host")    → "127.0.0.1"
        """
        keys = key.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
        return val if val is not None else default

    def set(self, key: str, value: Any):
        """支持 dot notation 设置配置（运行时修改，不写入文件）"""
        keys = key.split(".")
        d = self._config
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    # ==================== 快捷属性 ====================

    @property
    def general(self) -> dict:
        return self._config.get("general", {})

    @property
    def tools(self) -> dict:
        return self._config.get("tools", {})

    @property
    def database(self) -> dict:
        return self.tools.get("database", {})

    @property
    def cost_model(self) -> dict:
        return self._config.get("cost_model", {})

    @property
    def training(self) -> dict:
        return self._config.get("training", {})

    @property
    def evaluation(self) -> dict:
        return self._config.get("evaluation", {})

    # ==================== 工具方法 ====================

    def get_db_connection_params(self) -> dict:
        """返回 psycopg2.connect() 所需的参数"""
        db = self.database
        return {
            "host": db.get("host", "127.0.0.1"),
            "port": db.get("port", 5432),
            "user": db.get("user", "postgres"),
            "password": db.get("password", ""),
            "database": db.get("database", "postgres"),
        }

    def get_db_connection(self):
        """创建并返回一个 PG 连接"""
        import psycopg2
        return psycopg2.connect(**self.get_db_connection_params())

    def setup_logging(self):
        """根据配置初始化日志"""
        general = self.general
        log_dir = general.get("log_dir", "./logs")
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, general.get("log_level", "INFO")),
            format=general.get(
                "log_format",
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            ),
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    f"{log_dir}/db-opt-r1.log", encoding="utf-8"
                ),
            ],
        )

    def __repr__(self):
        return f"Config({self._path})"
