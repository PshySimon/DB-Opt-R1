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
│   │   └── scenarios/                  # 种子 + knob 配置 + 采集结果
│   └── synthesis/
│       ├── scenarios/                  # 场景生成 pipeline
│       └── mcts/                       # MCTS 轨迹合成
├── configs/
│   ├── config.yaml                     # 全局配置
│   ├── knob_space.yaml                 # 45 个可调 knob 定义
│   └── knob_effects.yaml              # Knob 效果知识库（瓶颈方向 + 效果描述）
└── scripts/
    ├── setup_env.sh                    # 环境准备
    └── collect_costmodel.sh            # Cost Model 数据采集
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
- 创建 benchmark 数据库 + pgbench 初始化
- 安装 Python 依赖

---

## 使用流程

### 1. Cost Model 数据采集

在真机上随机采样 knob 配置 → pgbench → 采集指标，输出 CSV。

```bash
# 默认后台执行（日志 + PID 保存到 logs/costmodel/）
bash scripts/collect_costmodel.sh --rounds 1500 --database benchmark --workload all

# 调试时前台执行
bash scripts/collect_costmodel.sh --rounds 10 --foreground
```

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

#### Step 0: 按瓶颈方向生成种子

基于 `knob_effects.yaml` 中的 8 个瓶颈方向（memory/optimizer/wal/vacuum/parallel/bgwriter/connections/locks），LLM 逐方向生成性能瓶颈场景描述。

```bash
python3 -m datasets.synthesis.scenarios.pipeline seeds \
    --effects configs/knob_effects.yaml \
    --knob-space configs/knob_space.yaml \
    --output datasets/data/scenarios/seeds.json \
    --count 100 --model gpt-5 \
    --api-key $OPENAI_API_KEY \
    --api-base $OPENAI_API_BASE
```

#### Step 1: LLM 生成 knob 配置

LLM 根据种子描述生成**可用但有优化空间**的 knob 配置。通过 `--cpu/--memory/--disk` 指定目标硬件。

```bash
# 4c8g SSD 机器
python3 -m datasets.synthesis.scenarios.pipeline generate \
    --seeds datasets/data/scenarios/seeds.json \
    --output datasets/data/scenarios/knob_configs_4c8g_ssd.json \
    --config configs/knob_space.yaml \
    --cpu 4 --memory 8 --disk SSD \
    --model gpt-5 --variants 5 --workers 5 \
    --api-key $OPENAI_API_KEY \
    --api-base $OPENAI_API_BASE

# 8c16g HDD 机器
python3 -m datasets.synthesis.scenarios.pipeline generate \
    --seeds datasets/data/scenarios/seeds.json \
    --output datasets/data/scenarios/knob_configs_8c16g_hdd.json \
    --config configs/knob_space.yaml \
    --cpu 8 --memory 16 --disk HDD \
    --model gpt-5 --variants 5 --workers 5 \
    --api-key $OPENAI_API_KEY \
    --api-base $OPENAI_API_BASE
```

#### Step 2: 真机采集（需要 PG）

将 knob 配置应用到 PG → pgbench → 采集完整指标（CPU/IO/等待事件/慢查询/日志）。

```bash
# 后台执行（日志 + PID 保存到 logs/scenarios/）
mkdir -p logs/scenarios
nohup python3 -u -m datasets.synthesis.scenarios.pipeline collect \
    --input datasets/data/scenarios/knob_configs_8c16g_hdd.json \
    --output datasets/data/scenarios/collected.json \
    --host 127.0.0.1 --port 5432 \
    --user postgres --database benchmark \
    > logs/scenarios/$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > logs/scenarios/running.pid

# 查看进度
tail -f logs/scenarios/*.log
# 停止采集
kill $(cat logs/scenarios/running.pid)
```

#### Step 3: MCTS 轨迹合成

基于采集的场景数据，MCTS 搜索最优调优轨迹，生成 SFT + 对比对数据。

```bash
python3 -m datasets.synthesis.mcts.run_search \
    --scenarios datasets/data/scenarios/collected.json \
    --knob-space configs/knob_space.yaml \
    --cost-model cost_model/saved/model.pkl \
    --output-dir datasets/data
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
