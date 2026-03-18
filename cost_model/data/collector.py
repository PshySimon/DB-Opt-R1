"""
db-opt-r1 数据采集工具

采集四大类特征：
1. 硬件特征（CPU、内存、磁盘）
2. 软件特征（PG 版本、OS、内核、文件系统）
3. Knob 配置（SHOW ALL）
4. DB 运行时指标（pg_stat 视图）
"""

import os
import json
import platform
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class HardwareCollector:
    """采集硬件特征"""

    def collect(self) -> dict:
        return {
            "cpu_count": os.cpu_count(),
            "cpu_model": self._get_cpu_model(),
            "cpu_freq_mhz": self._get_cpu_freq(),
            "total_memory_gb": self._get_memory_total(),
            "available_memory_gb": self._get_memory_available(),
            "disk_type": self._get_disk_type(),
            "disk_read_bandwidth_mbps": None,   # 需要 fio 测试，默认不采
            "disk_write_bandwidth_mbps": None,
            "disk_capacity_gb": self._get_disk_capacity(),
        }

    def _get_cpu_model(self) -> str:
        try:
            if platform.system() == "Linux":
                output = subprocess.check_output(
                    "grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2",
                    shell=True, text=True
                )
                return output.strip()
            elif platform.system() == "Darwin":
                output = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True
                )
                return output.strip()
        except Exception as e:
            logger.warning(f"获取 CPU 型号失败: {e}")
        return "unknown"

    def _get_cpu_freq(self) -> Optional[float]:
        try:
            if platform.system() == "Linux":
                output = subprocess.check_output(
                    "grep 'cpu MHz' /proc/cpuinfo | head -1 | cut -d: -f2",
                    shell=True, text=True
                )
                return round(float(output.strip()), 1)
            elif platform.system() == "Darwin":
                output = subprocess.check_output(
                    ["sysctl", "-n", "hw.cpufrequency"],
                    text=True
                )
                return round(int(output.strip()) / 1e6, 1)
        except Exception:
            pass
        return None

    def _get_memory_total(self) -> float:
        try:
            if platform.system() == "Linux":
                output = subprocess.check_output(
                    "grep MemTotal /proc/meminfo | awk '{print $2}'",
                    shell=True, text=True
                )
                return round(int(output.strip()) / 1024 / 1024, 1)
            elif platform.system() == "Darwin":
                output = subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"], text=True
                )
                return round(int(output.strip()) / 1024 / 1024 / 1024, 1)
        except Exception:
            pass
        return 0

    def _get_memory_available(self) -> float:
        try:
            if platform.system() == "Linux":
                output = subprocess.check_output(
                    "grep MemAvailable /proc/meminfo | awk '{print $2}'",
                    shell=True, text=True
                )
                return round(int(output.strip()) / 1024 / 1024, 1)
        except Exception:
            pass
        return 0

    def _get_disk_type(self) -> str:
        try:
            if platform.system() == "Linux":
                # rotational=0 表示 SSD, 1 表示 HDD
                output = subprocess.check_output(
                    "lsblk -d -o name,rota | grep -v NAME | head -1 | awk '{print $2}'",
                    shell=True, text=True
                )
                return "SSD" if output.strip() == "0" else "HDD"
        except Exception:
            pass
        return "unknown"

    def _get_disk_capacity(self) -> float:
        try:
            output = subprocess.check_output(
                "df -BG / | tail -1 | awk '{print $2}'",
                shell=True, text=True
            )
            return float(output.strip().replace("G", ""))
        except Exception:
            pass
        return 0


class SoftwareCollector:
    """采集软件特征"""

    def collect(self, pg_conn) -> dict:
        return {
            "os_type": platform.system(),
            "os_version": self._get_os_version(),
            "kernel_version": platform.release(),
            "filesystem": self._get_filesystem(),
            "huge_pages": self._get_huge_pages(),
            "pg_version": self._get_pg_version(pg_conn),
            "pg_encoding": self._get_pg_setting(pg_conn, "server_encoding"),
            "pg_max_connections": self._get_pg_setting(pg_conn, "max_connections"),
        }

    def _get_os_version(self) -> str:
        try:
            if platform.system() == "Linux":
                output = subprocess.check_output(
                    "grep PRETTY_NAME /etc/os-release | cut -d= -f2",
                    shell=True, text=True
                )
                return output.strip().strip('"')
            elif platform.system() == "Darwin":
                return platform.mac_ver()[0]
        except Exception:
            pass
        return platform.platform()

    def _get_filesystem(self) -> str:
        try:
            if platform.system() == "Linux":
                output = subprocess.check_output(
                    "df -T / | tail -1 | awk '{print $2}'",
                    shell=True, text=True
                )
                return output.strip()
        except Exception:
            pass
        return "unknown"

    def _get_huge_pages(self) -> str:
        try:
            if platform.system() == "Linux":
                output = subprocess.check_output(
                    "grep HugePages_Total /proc/meminfo | awk '{print $2}'",
                    shell=True, text=True
                )
                total = int(output.strip())
                return "on" if total > 0 else "off"
        except Exception:
            pass
        return "unknown"

    def _get_pg_version(self, conn) -> str:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version_str = cursor.fetchone()[0]
            cursor.close()
            # "PostgreSQL 16.2 on ..." → "16.2"
            return version_str.split()[1]
        except Exception as e:
            logger.warning(f"获取 PG 版本失败: {e}")
            return "unknown"

    def _get_pg_setting(self, conn, name: str) -> str:
        try:
            cursor = conn.cursor()
            cursor.execute(f"SHOW {name}")
            value = cursor.fetchone()[0]
            cursor.close()
            return value
        except Exception:
            return "unknown"


class KnobCollector:
    """采集 Knob 配置（SHOW ALL）"""

    def collect(self, pg_conn) -> dict:
        knobs = {}
        try:
            cursor = pg_conn.cursor()
            cursor.execute("SHOW ALL")
            for row in cursor.fetchall():
                name, value = row[0], row[1]
                knobs[name] = value
            cursor.close()
        except Exception as e:
            logger.error(f"采集 knob 失败: {e}")
        return knobs


class MetricsCollector:
    """采集 DB 运行时指标（pg_stat 视图）"""

    # 全局视图（单行）
    GLOBAL_VIEWS = [
        "pg_stat_bgwriter",
        "pg_stat_archiver",
    ]

    # 需要按 key 聚合的本地视图
    LOCAL_VIEWS = {
        "pg_stat_database": "datname",
        "pg_stat_database_conflicts": "datname",
        "pg_stat_user_tables": "relname",
        "pg_statio_user_tables": "relname",
        "pg_stat_user_indexes": "relname",
        "pg_statio_user_indexes": "relname",
    }

    def collect(self, pg_conn, database: str = None) -> dict:
        metrics = {
            "global": {},
            "local": {},
        }

        cursor = pg_conn.cursor()

        # 全局视图
        for view in self.GLOBAL_VIEWS:
            try:
                metrics["global"][view] = self._query_view(cursor, view)
            except Exception as e:
                logger.warning(f"采集 {view} 失败: {e}")

        # 本地视图
        for view, key_col in self.LOCAL_VIEWS.items():
            try:
                rows = self._query_view_grouped(cursor, view, key_col, database)
                metrics["local"][view] = rows
            except Exception as e:
                logger.warning(f"采集 {view} 失败: {e}")

        cursor.close()
        return metrics

    def _query_view(self, cursor, view: str) -> dict:
        cursor.execute(f"SELECT * FROM {view}")
        columns = [desc[0] for desc in cursor.description]
        row = cursor.fetchone()
        if row:
            return {col: str(val) for col, val in zip(columns, row)}
        return {}

    def _query_view_grouped(self, cursor, view: str, key_col: str,
                            database: str = None) -> dict:
        sql = f"SELECT * FROM {view}"
        if database and "database" in view:
            sql += f" WHERE {key_col} = '{database}'"
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        result = {}
        for row in cursor.fetchall():
            row_dict = {col: str(val) for col, val in zip(columns, row)}
            key = row_dict.get(key_col, "unknown")
            result[key] = row_dict
        return result


class DataCollector:
    """统一数据采集入口"""

    def __init__(self, pg_conn, database: str = None):
        self.pg_conn = pg_conn
        self.database = database
        self.hardware_collector = HardwareCollector()
        self.software_collector = SoftwareCollector()
        self.knob_collector = KnobCollector()
        self.metrics_collector = MetricsCollector()

    def collect_snapshot(self) -> dict:
        """采集一次完整快照"""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "hardware": self.hardware_collector.collect(),
            "software": self.software_collector.collect(self.pg_conn),
            "knobs": self.knob_collector.collect(self.pg_conn),
            "metrics": self.metrics_collector.collect(self.pg_conn, self.database),
        }
        return snapshot

    def collect_diff(self, before: dict, after: dict) -> dict:
        """计算两次快照的 metrics 差值"""
        diff = {
            "timestamp": after["timestamp"],
            "hardware": after["hardware"],
            "software": after["software"],
            "knobs": after["knobs"],
            "metrics_diff": self._diff_metrics(
                before["metrics"], after["metrics"]
            ),
        }
        return diff

    def _diff_metrics(self, before: dict, after: dict) -> dict:
        """对数值型 metrics 计算 after - before"""
        diff = {"global": {}, "local": {}}

        # 全局视图差值
        for view in before.get("global", {}):
            if view in after.get("global", {}):
                diff["global"][view] = self._diff_dict(
                    before["global"][view],
                    after["global"][view]
                )

        # 本地视图差值
        for view in before.get("local", {}):
            if view in after.get("local", {}):
                diff["local"][view] = {}
                for key in before["local"][view]:
                    if key in after["local"][view]:
                        diff["local"][view][key] = self._diff_dict(
                            before["local"][view][key],
                            after["local"][view][key]
                        )

        return diff

    def _diff_dict(self, before: dict, after: dict) -> dict:
        result = {}
        for key in after:
            try:
                val_after = float(after[key])
                val_before = float(before.get(key, 0))
                result[key] = val_after - val_before
            except (ValueError, TypeError):
                # 非数值字段保留 after 的值
                result[key] = after[key]
        return result

    def flatten_snapshot(self, snapshot: dict) -> dict:
        """将嵌套快照打平为一行（用于 CSV）"""
        flat = {}
        flat["timestamp"] = snapshot["timestamp"]

        # 硬件特征: hw_cpu_count, hw_total_memory_gb, ...
        for k, v in snapshot["hardware"].items():
            flat[f"hw_{k}"] = v

        # 软件特征: sw_os_type, sw_pg_version, ...
        for k, v in snapshot["software"].items():
            flat[f"sw_{k}"] = v

        # Knob 配置: knob_shared_buffers, knob_work_mem, ...
        for k, v in snapshot["knobs"].items():
            flat[f"knob_{k}"] = v

        # 全局 metrics: metric_pg_stat_bgwriter_buffers_alloc, ...
        # 兼容 collect_snapshot (metrics) 和 collect_diff (metrics_diff)
        metrics_data = snapshot.get("metrics") or snapshot.get("metrics_diff", {})
        for view, data in metrics_data.get("global", {}).items():
            for k, v in data.items():
                flat[f"metric_{view}_{k}"] = v

        # 本地 metrics: 聚合（求和）所有表/索引的数值
        for view, tables in metrics_data.get("local", {}).items():
            agg = {}
            for table_name, data in tables.items():
                for k, v in data.items():
                    try:
                        num = float(v)
                        agg[k] = agg.get(k, 0) + num
                    except (ValueError, TypeError):
                        pass
            for k, v in agg.items():
                flat[f"metric_{view}_sum_{k}"] = v

        return flat

    def save_json(self, data: dict, output_dir: str, filename: str = None):
        """保存原始快照到 JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{ts}.json"

        filepath = output_path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"JSON 已保存到 {filepath}")
        return str(filepath)

    def save_csv(self, flat_row: dict, output_dir: str,
                 filename: str = "dataset.csv"):
        """追加一行到 CSV 文件（自动对齐列）"""
        import csv

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename

        file_exists = filepath.exists() and filepath.stat().st_size > 0

        if file_exists:
            # 读取已有 header
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_columns = next(reader)
            # 合并：已有列 + 新列（如果有的话）
            new_cols = [k for k in flat_row.keys() if k not in existing_columns]
            all_columns = existing_columns + new_cols
        else:
            all_columns = list(flat_row.keys())

        # 值中的逗号和引号需要正确转义
        cleaned_row = {}
        for col in all_columns:
            val = flat_row.get(col, "")
            if val is None:
                val = ""
            cleaned_row[col] = str(val)

        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_columns,
                                    quoting=csv.QUOTE_ALL)
            if not file_exists:
                writer.writeheader()
            writer.writerow(cleaned_row)

        logger.info(f"CSV 已追加到 {filepath}")
        return str(filepath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DB 数据采集工具")
    parser.add_argument("--host", default="127.0.0.1", help="PG 主机")
    parser.add_argument("--port", type=int, default=5432, help="PG 端口")
    parser.add_argument("--user", default="postgres", help="PG 用户")
    parser.add_argument("--password", default="", help="PG 密码")
    parser.add_argument("--database", default="postgres", help="PG 数据库名")
    parser.add_argument("--output", default="./cost_model_data/raw", help="输出目录")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        import psycopg2
        conn = psycopg2.connect(
            host=args.host, port=args.port,
            user=args.user, password=args.password,
            database=args.database
        )
        collector = DataCollector(conn, database=args.database)
        snapshot = collector.collect_snapshot()

        # 保存 JSON 原始快照
        json_path = collector.save_json(snapshot, args.output)
        print(f"JSON 快照: {json_path}")

        # 打平并追加到 CSV
        flat = collector.flatten_snapshot(snapshot)
        csv_path = collector.save_csv(flat, args.output)
        print(f"CSV 数据集: {csv_path}")

        conn.close()
    except ImportError:
        print("请安装 psycopg2: pip install psycopg2-binary")
    except Exception as e:
        print(f"采集失败: {e}")
