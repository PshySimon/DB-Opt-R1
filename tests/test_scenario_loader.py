import unittest

from data_pipeline.synthesis.scenarios.loader import dedup_scenarios


def _item(name, tps, seq_write_bw=0.9, seq_write_mbps=450.0):
    return {
        "name": name,
        "knobs": {
            "shared_buffers": "4GB",
            "synchronous_commit": "off",
        },
        "workload": {
            "type": "write_heavy",
            "tps_current": tps,
        },
        "hardware": {
            "cpu_count": 8,
            "cpu_model": "Intel(R) Xeon",
            "total_memory_gb": 15.6,
            "disk_type": "HDD",
            "disk_capacity_gb": 196.0,
            "seq_write_bw_fio_mbps": seq_write_bw,
            "seq_write_p99_lat_us": 450,
            "rand_read_iops": 35000,
            "rand_read_mbps": 140.0,
            "seq_write_mbps": seq_write_mbps,
            "seq_read_mbps": 355.0,
            "mem_bw_gbps": 18.6,
        },
    }


class ScenarioLoaderDedupTest(unittest.TestCase):
    def test_dedup_keeps_same_knobs_when_io_profile_differs(self):
        items = [
            _item("slow_io", 5000.0, seq_write_bw=0.5, seq_write_mbps=450.0),
            _item("fast_io", 12000.0, seq_write_bw=1.1, seq_write_mbps=800.0),
        ]

        deduped = dedup_scenarios(items)

        self.assertEqual([x["name"] for x in deduped], ["slow_io", "fast_io"])

    def test_dedup_removes_conflicting_tps_when_io_profile_is_close(self):
        items = [
            _item("lower_tps", 5000.0, seq_write_bw=0.91, seq_write_mbps=451.0),
            _item("higher_tps", 7000.0, seq_write_bw=0.94, seq_write_mbps=454.0),
        ]

        deduped = dedup_scenarios(items)

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["name"], "higher_tps")


if __name__ == "__main__":
    unittest.main()
