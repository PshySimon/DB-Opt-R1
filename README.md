# DB-Opt-R1

基于 GRPO 强化学习训练 LLM Agent 进行 PostgreSQL 自动调优。

## 思路

```
                  ┌─────────────┐
                  │   LLM Agent │  ← GRPO 训练
                  └──────┬──────┘
                         │ 调用工具
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         观察工具     行动工具     验证工具
       (硬件/指标)   (set_knob)  (cost model)
              │          │          │
              └──────────┼──────────┘
                         ▼
                    模拟环境 / 真实 PG
```

1. **数据采集**：在真实 PG 上随机采样 knob 配置 + 跑 benchmark，收集性能数据
2. **Cost Model**：用采集的数据训练性能预测模型（LightGBM）
3. **SFT 数据合成**：MCTS 搜索最优调优轨迹，生成 SFT/DPO 训练数据
4. **GRPO 训练**：LLM Agent 通过工具与模拟环境交互，Cost Model 预测性能作为 reward
5. **评估**：在真实 PG 上验证 Agent 的调优效果

## 项目结构

```
├── core/                       # 核心框架
│   ├── config.py               # 统一配置管理
│   └── tool/                   # 工具框架（Tool 基类、ToolEnv、装饰器）
├── environment/
│   └── tools/                  # 11 个 DB 调优工具（real/simulated 双模式）
├── cost_model/
│   └── data/                   # 数据采集 pipeline
├── datasets/
│   ├── synthesis/mcts/         # MCTS 轨迹数据合成
│   └── data/                   # 生成的数据集
├── configs/
│   ├── config.yaml             # 全局配置
│   └── knob_space.yaml         # 45 个可调 knob 定义
├── scripts/
│   ├── setup_env.sh            # 环境准备（支持容器/VM）
│   └── collect_data.sh         # 数据采集
└── docs/                       # 设计文档
```

## 环境准备

**系统要求**：Ubuntu 22.04+，Python 3.8+

```bash
# 一键安装（自动检测 root/sudo、systemd/容器）
bash scripts/setup_env.sh
```

自动完成：
- 安装 PostgreSQL 16 + pgbench（国内优先系统源）
- 开启统计收集（`track_io_timing`）和慢查询日志
- 配置本地免密登录（Unix socket + TCP 127.0.0.1）
- 安装 Python 依赖

## 数据采集

**首次使用**必须加 `--init` 初始化 pgbench 测试表（只需一次）：

```bash
# 第一次：初始化 + 采集
bash scripts/collect_data.sh --init --rounds 1500 --database benchmark --workload all --background
```

之后再跑**无需** `--init`：

```bash
# 后续：直接采集
bash scripts/collect_data.sh --rounds 1500 --database benchmark --workload all --background
```

> ⚠️ 重建数据库或重装 PG 后需重新 `--init`

**负载类型**：

| 类型 | 说明 |
|------|------|
| `mixed` | TPC-B 读写混合（默认） |
| `read_only` | 只读查询 |
| `high_concurrency` | 64 并发连接 |
| `write_heavy` | 写密集 |
| `all` | 每种各跑一批（推荐） |

**后台运行**会输出日志路径和 PID：
```
PID: 12345 (已保存到 ./cost_model/data/raw/collect.pid)
查看进度: tail -f ./cost_model/data/raw/collect_*.log
停止采集: kill $(cat ./cost_model/data/raw/collect.pid)
```

**全部参数**：`bash scripts/collect_data.sh --help`

## 配置

编辑 `configs/config.yaml`：

```yaml
tools:
  database:
    host: "127.0.0.1"
    port: 5432
    user: "postgres"
    password: "postgres"
    database: "benchmark"
```

## 参考项目

- [Agent-R1](https://github.com/0russwest0/Agent-R1) — 工具框架参考
- [DBTune](https://github.com/PKU-DAIR/DBTune) — DB 调优方法参考
- [GPTuner](https://github.com/SolidLao/GPTuner) — LLM + 代理模型调优

## License

MIT
