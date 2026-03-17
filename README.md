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
2. **Cost Model**：用采集的数据训练性能预测模型
3. **GRPO 训练**：LLM Agent 通过工具与模拟环境交互，Cost Model 预测性能作为 reward
4. **评估**：在真实 PG 上验证 Agent 的调优效果

## 项目结构

```
├── core/                   # 核心框架
│   ├── config.py           # 统一配置管理
│   └── tool/               # 工具框架（Tool 基类、ToolEnv、装饰器）
├── environment/
│   └── tools/              # 11 个 DB 调优工具（real/simulated 双模式）
├── cost_model/
│   └── data/               # 数据采集 pipeline
├── configs/
│   ├── config.yaml         # 全局配置
│   └── knob_space.yaml     # 45 个可调 knob 定义
├── scripts/
│   ├── setup_env.sh        # 环境准备
│   └── collect_data.sh     # 数据采集
└── docs/                   # 设计文档
```

## 环境准备

**系统要求**：Ubuntu，Python 3.8+

```bash
# 1. 安装 PostgreSQL 16 + 系统依赖
sudo bash scripts/setup_env.sh

# 2. 安装 Python 依赖
pip install -r requirements.txt
```

`setup_env.sh` 会自动完成：
- 安装 PostgreSQL 16 + pgbench
- 开启统计收集（`track_io_timing`）和慢查询日志
- 创建 benchmark 数据库
- 配置本地免密登录

**自定义 PG 版本**：
```bash
PG_VERSION=15 sudo bash scripts/setup_env.sh
```

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

## 数据采集

```bash
# 初始化 benchmark 数据 + 采集 100 轮
bash scripts/collect_data.sh --init --rounds 100 --database benchmark
```

## 参考项目

- [Agent-R1](https://github.com/0russwest0/Agent-R1) — 工具框架参考
- [DBTune](https://github.com/PKU-DAIR/DBTune) — DB 调优方法参考
- [GPTuner](https://github.com/SolidLao/GPTuner) — LLM + 代理模型调优

## License

MIT
