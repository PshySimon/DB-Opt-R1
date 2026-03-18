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
├── core/                               # 核心框架
│   ├── config.py                       # 统一配置管理
│   ├── tool/                           # 工具框架（Tool 基类、ToolEnv、装饰器）
│   └── db/                             # PG 公共基础设施
│       ├── knob_space.py               # Knob 搜索空间 + 校验
│       ├── collector.py                # 数据采集器
│       ├── pg_configurator.py          # PG 配置管理
│       └── benchmark_runner.py         # Benchmark 运行器
├── environment/
│   └── tools/                          # 12 个 DB 调优工具（real/simulated 双模式）
├── cost_model/
│   ├── data/                           # Cost Model 数据采集 pipeline
│   ├── train.py                        # 训练脚本
│   └── preprocess.py                   # 特征工程
├── datasets/
│   ├── data/                           # 所有数据集统一存放
│   │   ├── cost_model/                 # Cost Model 采集数据
│   │   ├── scenario_seeds/             # 场景种子描述
│   │   └── scenarios/                  # 场景 knob 配置 + 采集结果
│   └── synthesis/
│       ├── scenarios/                  # 场景生成 pipeline
│       └── mcts/                       # MCTS 轨迹合成
├── configs/
│   ├── config.yaml                     # 全局配置
│   └── knob_space.yaml                 # 45 个可调 knob 定义
└── scripts/
    ├── setup_env.sh                    # 环境准备
    └── collect_data.sh                 # 数据采集
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

---

## 使用流程

### 1. Cost Model 数据采集

在真机上随机采样 knob 配置 → pgbench → 采集指标，输出 CSV。

```bash
# 首次：初始化 pgbench 测试表 + 采集
bash scripts/collect_data.sh --init --rounds 1500 --database benchmark --workload all --background

# 后续：直接采集（无需 --init）
bash scripts/collect_data.sh --rounds 1500 --database benchmark --workload all --background
```

或直接调用 Python pipeline：

```bash
python3 -m cost_model.data.pipeline \
    --config configs/knob_space.yaml \
    --output datasets/data/cost_model \
    --rounds 1500 --sampling lhs \
    --database benchmark
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

### 2. Cost Model 训练

```bash
python3 -m cost_model.train \
    --data datasets/data/cost_model/dataset.csv \
    --save-dir cost_model/saved
```

### 3. SFT 数据合成（场景驱动）

三步流程，每步独立，支持断点续跑：

#### Step 0: 生成种子场景描述（可选，扩充种子库）

LLM 根据 knob_space 自动生成多样化的故障/调优场景描述。

```bash
python3 -m datasets.synthesis.scenarios.pipeline seeds \
    --config configs/knob_space.yaml \
    --output datasets/data/scenario_seeds/seeds.json \
    --count 50 \
    --model gpt-4 \
    --api-key $OPENAI_API_KEY \
    --api-base $OPENAI_API_BASE
```

#### Step 1: 生成 knob 配置（不需要 PG）

LLM 根据种子描述生成对应的 knob 配置，经 `KnobSpace.validate()` 校验。

```bash
python3 -m datasets.synthesis.scenarios.pipeline generate \
    --seeds datasets/data/scenario_seeds/seeds.json \
    --output datasets/data/scenarios/knob_configs.json \
    --config configs/knob_space.yaml \
    --model gpt-4 --variants 3 --workers 5
```

#### Step 2: 真机采集（需要 PG）

将 knob 配置应用到 PG → pgbench → 采集完整指标（CPU/IO/等待事件/慢查询/日志）。

```bash
python3 -m datasets.synthesis.scenarios.pipeline collect \
    --input datasets/data/scenarios/knob_configs.json \
    --output datasets/data/scenarios/collected.json \
    --host 127.0.0.1 --port 5432 \
    --user postgres --database benchmark
```

#### Step 3: MCTS 轨迹合成

基于采集的场景数据，MCTS 搜索最优调优轨迹，生成 SFT + 对比对数据。

```bash
# 使用场景数据（新模式）
python3 -m datasets.synthesis.mcts.run_search \
    --scenarios datasets/data/scenarios/collected.json \
    --knob-space configs/knob_space.yaml \
    --cost-model cost_model/saved/model.pkl \
    --output-dir datasets/data

# 使用 CSV 数据（旧模式，仍兼容）
python3 -m datasets.synthesis.mcts.run_search \
    --dataset datasets/data/cost_model/dataset.csv \
    --knob-space configs/knob_space.yaml \
    --cost-model cost_model/saved/model.pkl
```

输出：
- `datasets/data/sft_data.jsonl` — SFT 训练数据
- `datasets/data/contrastive_pairs.jsonl` — DPO/对比学习数据

---

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
