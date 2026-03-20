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
│   ├── knob_generator.py               # Knob 随机采样（random/near_default/LHS/mixed）
│   ├── train.py                        # 训练脚本
│   ├── model.py                        # 模型定义
│   └── preprocess.py                   # 特征工程（支持 CSV/JSON 输入）
├── datasets/
│   ├── data/scenarios/                 # 种子 + knob 配置 + 采集结果（统一 JSON）
│   └── synthesis/
│       ├── scenarios/                  # 场景生成 + 采集 pipeline
│       └── mcts/                       # MCTS 轨迹合成
├── configs/
│   ├── config.yaml                     # 全局配置
│   ├── knob_space.yaml                 # 45 个可调 knob 定义
│   └── knob_effects.yaml              # Knob 效果知识库（瓶颈方向 + 效果描述）
└── scripts/
    └── setup_env.sh                    # 环境准备
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

### 统一数据 Pipeline（Step 0 → 5）

所有数据采集、Cost Model 训练、MCTS 轨迹合成合并为一条流水线：

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

#### Step 2: 随机采样 knob 配置（Cost Model 数据）

纯本地生成，不需要 LLM。采样策略 `mixed` = 40% 随机 + 40% 默认值附近 + 20% LHS。

```bash
python3 -m datasets.synthesis.scenarios.pipeline random-sample \
    --knob-space configs/knob_space.yaml \
    --output datasets/data/scenarios/knob_configs_random.json \
    --count 200 --strategy mixed
```

#### Step 3: 统一真机采集（需要 PG）

`collect` 支持 glob，自动合并所有 `knob_configs_*.json`（LLM 生成 + 随机采样），一次采集。

```bash
# 后台执行
mkdir -p logs/scenarios
nohup python3 -m datasets.synthesis.scenarios.pipeline collect \
    --input "datasets/data/scenarios/knob_configs_*.json" \
    --output datasets/data/scenarios/collected.json \
    --host 127.0.0.1 --port 5432 \
    --user postgres --database benchmark \
    > logs/scenarios/$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > logs/scenarios/running.pid
```

数据带 `source` 标签：`llm_generated`（MCTS + Cost Model）、`random_sampled`（仅 Cost Model）。

#### Step 4: Cost Model 训练

使用全量 `collected.json` 数据训练 Cost Model。

```bash
python3 -m cost_model.train \
    --data datasets/data/scenarios/collected.json \
    --output cost_model/checkpoints/v1
```

#### Step 5: MCTS 轨迹合成

基于 `source=llm_generated` 场景 + Cost Model，搜索最优调优轨迹。每次运行输出到带时间戳的子目录。

```bash
python3 -m datasets.synthesis.mcts.run_search \
    --scenarios datasets/data/scenarios/ \
    --knob-space configs/knob_space.yaml \
    --cost-model cost_model/checkpoints/v3 \
    --output-dir datasets/data \
    --model gpt-5 \
    --api-key $OPENAI_API_KEY \
    --api-base $OPENAI_API_BASE \
    --parallel 4 \
    --num-workers 2 \
    --simulations 5
```

**参数说明**：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--scenarios` | 场景数据源（目录或文件，自动过滤 `source=llm_generated`） | - |
| `--num-envs` | 搜索环境数（0=全量） | 0 |
| `--parallel` | 多环境并行数（同时搜索 N 个场景） | 1 |
| `--num-workers` | 单棵树内并发 simulation 线程数 | 1 |
| `--simulations` | 每棵树 MCTS 迭代次数 | 5 |

输出（在 `datasets/data/run_YYYYMMDD_HHMMSS/` 下）：
- `sft_trajectories.jsonl` — SFT 训练数据
- `contrastive_pairs.jsonl` — DPO/对比学习数据
- `mcts_trees/` — 搜索树 debug 文件

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
