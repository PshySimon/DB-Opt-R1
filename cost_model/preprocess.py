"""
Cost Model 特征预处理 Pipeline
原始 CSV / JSON → 特征筛选 → 单位解析 → 缺失填充 → 变换 → 编码 → X, y
"""

import re
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ===== 单位解析 =====

def parse_memory_to_mb(val) -> float:
    """解析 PG 内存值为 MB: '128MB' → 128.0, '2GB' → 2048.0"""
    if isinstance(val, (int, float)):
        return float(val)
    if not val or pd.isna(val):
        return np.nan

    val = str(val).strip()
    m = re.match(r'^(-?[\d.]+)\s*(kB|MB|GB|TB|B)$', val, re.IGNORECASE)
    if m:
        num = float(m.group(1))
        unit = m.group(2).upper()
        units = {"B": 1/1048576, "KB": 1/1024, "MB": 1, "GB": 1024, "TB": 1048576}
        return num * units[unit]

    try:
        return float(val)
    except ValueError:
        return np.nan


def parse_time_to_ms(val) -> float:
    """解析 PG 时间值为 ms: '200ms' → 200.0, '5min' → 300000.0"""
    if isinstance(val, (int, float)):
        return float(val)
    if not val or pd.isna(val):
        return np.nan

    val = str(val).strip()
    m = re.match(r'^(-?[\d.]+)\s*(us|ms|s|min|h|d)$', val, re.IGNORECASE)
    if m:
        num = float(m.group(1))
        unit = m.group(2).lower()
        units = {"us": 0.001, "ms": 1, "s": 1000, "min": 60000, "h": 3600000, "d": 86400000}
        return num * units[unit]

    try:
        return float(val)
    except ValueError:
        return np.nan


def parse_bool(val) -> float:
    """解析布尔: 'on'→1, 'off'→0"""
    if isinstance(val, (int, float)):
        return float(val)
    if not val or pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ("on", "true", "yes", "1"):
        return 1.0
    if s in ("off", "false", "no", "0"):
        return 0.0
    return np.nan


class KnobPreprocessor:
    """
    Knob 特征预处理器

    用法：
        prep = KnobPreprocessor("configs/knob_space.yaml")
        X, y, meta = prep.fit_transform("datasets/data/scenarios/collected.json")
        prep.save("cost_model/checkpoints/v1/")

        # 推理
        prep = KnobPreprocessor.load("cost_model/checkpoints/v1/")
        x = prep.transform({"shared_buffers": "2GB", "work_mem": "64MB", ...})
    """

    # Knob 按类型分组（来自 knob_space.yaml 的 type 字段）
    MEMORY_KNOBS = [
        "shared_buffers", "work_mem", "effective_cache_size",
        "maintenance_work_mem", "wal_buffers", "temp_buffers",
        "max_wal_size", "min_wal_size",
    ]
    MEMORY_KNOBS_SMALL = ["max_stack_depth"]  # 范围小，不需 log

    TIME_KNOBS = [
        "autovacuum_naptime", "bgwriter_delay", "checkpoint_timeout",
        "lock_timeout", "deadlock_timeout", "idle_in_transaction_session_timeout",
    ]

    LOG_KNOBS = [
        "commit_delay", "autovacuum_analyze_threshold",
        "autovacuum_vacuum_threshold", "max_connections",
    ]
    LOG1P_KNOBS = [
        "effective_io_concurrency", "bgwriter_lru_maxpages",
    ]

    BOOL_KNOBS = [
        "autovacuum", "track_activities", "track_counts", "wal_compression",
    ]
    ENUM_KNOBS = {
        "synchronous_commit": ["off", "local", "on", "remote_write"],
    }

    # 直接使用原值的数值 knob
    PASSTHROUGH_KNOBS = [
        "random_page_cost", "seq_page_cost", "cpu_tuple_cost",
        "cpu_index_tuple_cost", "cpu_operator_cost",
        "checkpoint_completion_target", "bgwriter_lru_multiplier",
        "autovacuum_vacuum_scale_factor", "autovacuum_analyze_scale_factor",
        "default_statistics_target", "from_collapse_limit",
        "join_collapse_limit", "geqo_threshold", "autovacuum_max_workers",
        "max_parallel_workers", "max_parallel_workers_per_gather",
        "max_parallel_maintenance_workers", "superuser_reserved_connections",
        "max_wal_senders", "wal_sender_timeout",
    ]

    HW_NUMERIC = [
        "hw_cpu_count", "hw_cpu_freq_mhz", "hw_total_memory_gb",
        "hw_available_memory_gb", "hw_disk_capacity_gb",
        # IO 基准特征（_load_json 展平 hardware 字段时自动加 hw_ 前缀，字典里的 key 不带 hw_）
        "hw_seq_write_mbps",        # dd conv=fdatasync 顺序写 (MB/s)
        "hw_seq_read_mbps",         # dd 顺序读 (MB/s)
        "hw_rand_read_iops",        # fio randread 4K IOPS
        "hw_rand_read_mbps",        # fio randread 带宽 (MB/s)
        "hw_mem_bw_gbps",           # dd /dev/zero 内存带宽 (GB/s)
        "hw_seq_write_bw_fio_mbps",# fio seqwrite 带宽 (MB/s)
        "hw_seq_write_p99_lat_us", # fio seqwrite p99 延迟 (us)
    ]

    # 删除的列
    DROP_PREFIXES = ["sw_", "metric_"]
    DROP_EXACT = [
        "timestamp", "error",
        "knob_shared_memory_size", "knob_shared_memory_size_in_huge_pages",
        "hw_cpu_model",
    ]

    def __init__(self, knob_space_path: str = None):
        self.knob_defaults = {}
        self.feature_names = []
        self.varying_knobs = []
        self._fitted = False

        if knob_space_path:
            self._load_knob_space(knob_space_path)

    def _load_knob_space(self, path: str):
        """从 knob_space.yaml 加载默认值"""
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        for name, spec in config.get("knobs", {}).items():
            self.knob_defaults[name] = spec.get("default")

    def fit_transform(self, data_path: str):
        """训练时：加载数据（CSV 或 JSON），拟合 + 变换，返回 X, y, meta"""
        df = self._load_data(data_path)
        logger.info(f"  原始: {df.shape[0]} 行 × {df.shape[1]} 列")

        # 保存 status、workload、source 信息
        meta = pd.DataFrame({
            "status": df.get("status", "success"),
            "workload": df.get("workload", "mixed"),
            "source": df.get("source", "unknown"),
        })

        # 1. 找变化的 knob（用成功数据判断）
        success_mask = df["status"] == "success"
        knob_cols = [c for c in df.columns if c.startswith("knob_")]
        self.varying_knobs = []
        for col in knob_cols:
            vals = df.loc[success_mask, col].dropna().unique()
            if len(vals) > 1:
                name = col.replace("knob_", "")
                if name not in ("shared_memory_size", "shared_memory_size_in_huge_pages"):
                    self.varying_knobs.append(name)

        logger.info(f"  变化的 knob: {len(self.varying_knobs)} 个")

        # 2. 构建特征 DataFrame
        X = self._build_features(df)

        # 3. 目标变量
        y = np.log1p(pd.to_numeric(df["tps"], errors="coerce").fillna(0))

        self.feature_names = list(X.columns)
        self._fitted = True

        logger.info(f"  最终特征: {X.shape[1]} 个")
        logger.info(f"  样本: {X.shape[0]} (成功 {success_mask.sum()}, 失败 {(~success_mask).sum()})")

        return X, y, meta

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """加载数据：自动检测 CSV、单 JSON 或目录（collected_*.json）"""
        import os
        path = str(data_path)
        logger.info(f"加载数据: {path}")

        if os.path.isdir(path):
            # 目录：用 loader 统一加载 + 去重所有 collected_*.json
            from datasets.synthesis.scenarios.loader import load_scenario_files
            items = load_scenario_files(path, logger=logger)
            return self._items_to_df(items)
        elif path.endswith(".json"):
            return self._load_json(path)
        else:
            return pd.read_csv(path, on_bad_lines="skip")

    def _items_to_df(self, items: list) -> pd.DataFrame:
        """将 raw JSON dict list 展平为 DataFrame（knob_* / hw_* / workload / tps / source）"""
        rows = []
        for item in items:
            row = {}
            for k, v in item.get("knobs", {}).items():
                row[f"knob_{k}"] = v
            for k, v in item.get("hardware", {}).items():
                row[f"hw_{k}"] = v
            wl = item.get("workload", {})
            if isinstance(wl, dict):
                row["workload"] = wl.get("type", "mixed")
                row["tps"] = wl.get("tps_current", 0)
                row["latency_avg"] = wl.get("latency_avg_ms", 0)
            else:
                row["workload"] = str(wl)
                row["tps"] = 0
            row["status"] = "success"
            row["source"] = item.get("source", "unknown")
            rows.append(row)
        df = pd.DataFrame(rows)
        logger.info(f"  展平: {len(rows)} 条，{len(df.columns)} 列")
        return df

    def _load_json(self, json_path: str) -> pd.DataFrame:
        """加载单个 JSON 文件，去重后展平为 DataFrame"""
        import json as json_mod
        from datasets.synthesis.scenarios.loader import dedup_scenarios
        with open(json_path, "r", encoding="utf-8") as f:
            items = json_mod.load(f)
        items = dedup_scenarios(items, fname=json_path, logger=logger)
        return self._items_to_df(items)

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """从原始 df 提取并变换特征"""
        features = {}

        # ---- 内存 knob (log2) ----
        for name in self.MEMORY_KNOBS:
            col = f"knob_{name}"
            if col in df.columns:
                vals = df[col].apply(parse_memory_to_mb)
                default_mb = parse_memory_to_mb(self.knob_defaults.get(name, 0))
                vals = vals.fillna(default_mb)
                features[name] = np.log2(vals.clip(lower=1))

        # ---- 内存小范围 (原值) ----
        for name in self.MEMORY_KNOBS_SMALL:
            col = f"knob_{name}"
            if col in df.columns:
                vals = df[col].apply(parse_memory_to_mb)
                default_mb = parse_memory_to_mb(self.knob_defaults.get(name, 0))
                features[name] = vals.fillna(default_mb)

        # ---- 时间 knob (log10) ----
        for name in self.TIME_KNOBS:
            col = f"knob_{name}"
            if col in df.columns:
                vals = df[col].apply(parse_time_to_ms)
                default_ms = parse_time_to_ms(self.knob_defaults.get(name, 0))
                vals = vals.fillna(default_ms)
                features[name] = np.log10(vals.clip(lower=1))

        # ---- 数值 log knob ----
        for name in self.LOG_KNOBS:
            col = f"knob_{name}"
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                default_val = self.knob_defaults.get(name, 0)
                vals = vals.fillna(default_val)
                features[name] = np.log10(vals.clip(lower=1))

        # ---- 数值 log1p knob (含 0) ----
        for name in self.LOG1P_KNOBS:
            col = f"knob_{name}"
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                default_val = self.knob_defaults.get(name, 0)
                vals = vals.fillna(default_val)
                features[name] = np.log1p(vals)

        # ---- 直接使用的数值 knob ----
        for name in self.PASSTHROUGH_KNOBS:
            col = f"knob_{name}"
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                default_val = self.knob_defaults.get(name, 0)
                features[name] = vals.fillna(default_val)

        # ---- 布尔 knob ----
        for name in self.BOOL_KNOBS:
            col = f"knob_{name}"
            if col in df.columns:
                features[name] = df[col].apply(parse_bool).fillna(0)

        # ---- 枚举 knob ----
        for name, categories in self.ENUM_KNOBS.items():
            col = f"knob_{name}"
            if col in df.columns:
                default_val = self.knob_defaults.get(name, categories[0])
                vals = df[col].fillna(default_val)
                cat_map = {v: i for i, v in enumerate(categories)}
                features[name] = vals.map(cat_map).fillna(0).astype(float)

        # ---- 硬件特征（含 IO 基准特征，缺失时补 0 保证维度一致）----
        for col in self.HW_NUMERIC:
            if col in df.columns:
                features[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                features[col] = 0.0  # 旧数据无 IO 字段时补 0，维度保持一致

        # 磁盘类型
        if "hw_disk_type" in df.columns:
            features["hw_disk_type"] = (df["hw_disk_type"] == "SSD").astype(float)

        # ---- 负载 one-hot ----
        if "workload" in df.columns:
            for wl in ["mixed", "read_only", "high_concurrency", "write_heavy"]:
                features[f"workload_{wl}"] = (df["workload"] == wl).astype(float)

        # ---- 比值交互特征 ----
        # ---- 比值交互特征 ----
        # 用原始 MB 值计算（从已解析的 log2 值还原，或直接从 df 解析）
        if "hw_total_memory_gb" in df.columns:
            total_mem_gb = pd.to_numeric(df["hw_total_memory_gb"], errors="coerce").fillna(16)
        else:
            total_mem_gb = pd.Series(16, index=df.index)
        total_mem_mb = total_mem_gb * 1024

        if "knob_shared_buffers" in df.columns:
            sb_mb = df["knob_shared_buffers"].apply(parse_memory_to_mb).fillna(
                parse_memory_to_mb(self.knob_defaults.get("shared_buffers", "128MB")))
            features["ratio_shared_buffers_mem"] = sb_mb / total_mem_mb.clip(lower=1)

            if "knob_effective_cache_size" in df.columns:
                ec_mb = df["knob_effective_cache_size"].apply(parse_memory_to_mb).fillna(
                    parse_memory_to_mb(self.knob_defaults.get("effective_cache_size", "4GB")))
                features["ratio_cache_mem"] = ec_mb / total_mem_mb.clip(lower=1)
                features["ratio_cache_sb"] = ec_mb / sb_mb.clip(lower=1)

        if "knob_work_mem" in df.columns and "knob_max_connections" in df.columns:
            wm_mb = df["knob_work_mem"].apply(parse_memory_to_mb).fillna(
                parse_memory_to_mb(self.knob_defaults.get("work_mem", "4MB")))
            mc = pd.to_numeric(df["knob_max_connections"], errors="coerce").fillna(100)
            features["mem_pressure"] = np.log1p(wm_mb * mc / total_mem_mb.clip(lower=1))

        if "knob_max_parallel_workers_per_gather" in df.columns:
            pw = pd.to_numeric(df["knob_max_parallel_workers_per_gather"], errors="coerce").fillna(2)
            if "hw_cpu_count" in df.columns:
                cpu_count = pd.to_numeric(df["hw_cpu_count"], errors="coerce").fillna(8)
            else:
                cpu_count = pd.Series(8, index=df.index)
            features["ratio_parallel_cpu"] = pw / cpu_count.clip(lower=1)

        if "knob_max_wal_size" in df.columns and "knob_shared_buffers" in df.columns:
            wal_mb = df["knob_max_wal_size"].apply(parse_memory_to_mb).fillna(
                parse_memory_to_mb(self.knob_defaults.get("max_wal_size", "1GB")))
            sb_mb2 = df["knob_shared_buffers"].apply(parse_memory_to_mb).fillna(
                parse_memory_to_mb(self.knob_defaults.get("shared_buffers", "128MB")))
            features["ratio_wal_sb"] = wal_mb / sb_mb2.clip(lower=1)

        # ---- workload × top-knob 交叉特征 ----
        top_knobs_for_cross = ["synchronous_commit", "commit_delay", "shared_buffers"]
        if "workload" in df.columns:
            for knob_name in top_knobs_for_cross:
                if knob_name in features:
                    for wl in ["read_only", "write_heavy"]:
                        wl_mask = features.get(f"workload_{wl}", pd.Series(0, index=df.index))
                        features[f"cross_{wl}_{knob_name}"] = features[knob_name] * wl_mask

        # ---- 悬崖阈值特征 ----
        # 数据分析显示: commit_delay 和 synchronous_commit 对写 workload 有悬崖效应
        if "workload" in df.columns:
            is_write = (
                (df["workload"] == "write_heavy") |
                (df["workload"] == "mixed") |
                (df["workload"] == "high_concurrency")
            ).astype(float)

            # commit_delay 阈值 × 写 workload（cd>=5000 时写TPS降2-3倍）
            if "knob_commit_delay" in df.columns:
                cd_raw = pd.to_numeric(df["knob_commit_delay"], errors="coerce").fillna(0)
                cd_high = (cd_raw >= 5000).astype(float)
                cd_very_high = (cd_raw >= 50000).astype(float)
                features["cliff_cd_high_write"] = cd_high * is_write
                features["cliff_cd_vhigh_write"] = cd_very_high * is_write
                # commit_delay 对 read_only 无影响，单独标记
                is_read = features.get("workload_read_only", pd.Series(0, index=df.index))
                features["cliff_cd_high_read"] = cd_high * is_read

            # synchronous_commit=off × 写 workload（off时写TPS翻2-3.5倍）
            if "synchronous_commit" in features:
                sync_off = (features["synchronous_commit"] == 0).astype(float)
                features["cliff_sync_off_write"] = sync_off * is_write

        # ---- Top-10 Knob 两两交叉特征 (乘积) ----
        from itertools import combinations
        top_10_knobs = [
            'commit_delay', 'synchronous_commit', 'max_connections', 'lock_timeout',
            'checkpoint_completion_target', 'shared_buffers', 'effective_io_concurrency',
            'seq_page_cost', 'autovacuum_vacuum_threshold', 'random_page_cost'
        ]
        for a, b in combinations(top_10_knobs, 2):
            if a in features and b in features:
                features[f"x_{a}_x_{b}"] = features[a] * features[b]

        result = pd.DataFrame(features)

        # 处理残余 NaN
        result = result.fillna(0)

        return result

    def transform(self, knob_config: dict, hw_info: dict = None) -> np.ndarray:
        """
        推理时：将单条 knob 配置转为模型输入

        Args:
            knob_config: {"shared_buffers": "2GB", "work_mem": "64MB", ...}
            hw_info: {"cpu_count": 8, "total_memory_gb": 16, ...}, 可选
        """
        assert self._fitted, "预处理器未拟合，请先 fit_transform 或 load"

        # 硬件特征是区分三台机器 TPS 差异（最高 10 倍）的关键
        # hw_info 缺失时所有 hw_* 特征为 0，模型会收到训练集从未出现的分布，预测结果不可信
        if not hw_info:
            import warnings
            warnings.warn(
                "hw_info 未传入，所有硬件特征（rand_read_iops/mem_bw_gbps 等）将为 0，"
                "模型预测结果不可信，请确保传入正确的硬件信息。",
                UserWarning, stacklevel=2
            )

        # 构建单行 DataFrame
        row = {}
        for name in self.knob_defaults:
            row[f"knob_{name}"] = knob_config.get(name, self.knob_defaults[name])

        # 负载
        row["workload"] = knob_config.get("workload", "mixed")
        row["status"] = "success"

        # 硬件
        if hw_info:
            for k, v in hw_info.items():
                row[f"hw_{k}"] = v

        df = pd.DataFrame([row])
        X = self._build_features(df)

        # 对齐列顺序
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        return X.values[0]

    def save(self, output_dir: str):
        """保存预处理器状态"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        state = {
            "knob_defaults": self.knob_defaults,
            "feature_names": self.feature_names,
            "varying_knobs": self.varying_knobs,
        }
        with open(output_path / "preprocessor.pkl", "wb") as f:
            pickle.dump(state, f)

        with open(output_path / "feature_names.json", "w") as f:
            json.dump(self.feature_names, f, indent=2)

        logger.info(f"预处理器已保存到 {output_path}")

    @classmethod
    def load(cls, checkpoint_dir: str) -> "KnobPreprocessor":
        """加载预处理器"""
        path = Path(checkpoint_dir)
        with open(path / "preprocessor.pkl", "rb") as f:
            state = pickle.load(f)

        prep = cls()
        prep.knob_defaults = state["knob_defaults"]
        prep.feature_names = state["feature_names"]
        prep.varying_knobs = state["varying_knobs"]
        prep._fitted = True
        return prep


def load_dataset(data_path: str, knob_space_path: str):
    """快捷函数: CSV/JSON + knob_space → X, y, meta, preprocessor"""
    prep = KnobPreprocessor(knob_space_path)
    X, y, meta = prep.fit_transform(data_path)
    return X, y, meta, prep
