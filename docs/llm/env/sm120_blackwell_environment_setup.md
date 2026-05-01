# SM120 Blackwell 环境准备记录

本文记录 RTX PRO 6000 Blackwell Server Edition（`sm_120`）上 DB-Opt-R1 的训练与推理环境准备方式。目标不是追最新包，而是得到可复现、可冒烟、后续能写进环境准备脚本的版本组合。

## 机器与路径

| 项 | 值 |
|----|----|
| 项目目录 | `/root/autodl-tmp/DB-Opt-R1` |
| 基座模型 | `/root/autodl-tmp/models/Qwen3-8B` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | `(12, 0)` / `sm_120` |
| Driver 观测值 | `595.58.03` |
| Driver CUDA 观测值 | `13.2` |

基础检查：

```bash
nvidia-smi

python - <<'PY'
import torch
print("torch =", torch.__version__, torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
print("device =", torch.cuda.get_device_name(0))
print("capability =", torch.cuda.get_device_capability(0))
PY
```

## 结论摘要

| 场景 | 推荐环境 | 状态 |
|------|----------|------|
| LLaMA-Factory 训练 smoke | `torch==2.8.0+cu128` + `flash-attn==2.8.3` | 已通过 LoRA + FA2 1 step |
| 单卡 full SFT smoke | 不推荐 | Qwen3-8B full + optimizer 在 96G 单卡仍会 OOM |
| vLLM 默认推理 smoke | `vllm==0.11.0` + `torch==2.8.0+cu128` 可跑 | 只能作为可运行性兜底，不是性能目标 |
| vLLM + FlashInfer attention | `vllm==0.20.0` + `torch==2.11.0+cu130` + FlashInfer `0.6.9` 三件套 | 已通过 |

关键规则：

| 规则 | 原因 |
|------|------|
| 不使用 `uv` | 项目约定禁止；后续脚本全部用 `pip` |
| FlashInfer on `sm_120` 不走 `cu128` | FlashInfer 报错 `SM 12.x requires CUDA >= 12.9` |
| FlashInfer 三件套必须同 release | `flashinfer-python` / `flashinfer-cubin` / `flashinfer-jit-cache` 版本不一致会 import 失败 |
| `VLLM_ATTENTION_BACKEND=FLASHINFER` 对 vLLM 0.20 无效 | vLLM 会提示 unknown env var；必须用 Python API 的 `AttentionConfig` |
| vLLM spawn 不用 heredoc | `python - <<'PY'` 会导致子进程找 `/path/<stdin>` 失败 |

## 训练环境：LLaMA-Factory + FlashAttention-2

训练环境使用 CUDA 12.8 栈即可：

```text
torch                 2.8.0+cu128
transformers          4.52.4
datasets              3.6.0
accelerate            1.7.0
deepspeed             0.16.9
llamafactory          0.9.3
flash-attn            2.8.3
```

FlashAttention-2 使用预编译 wheel，不要默认走源码编译：

```bash
python -m pip install -U packaging ninja

python -m pip install --no-cache-dir \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

验证：

```bash
python - <<'PY'
import torch
import flash_attn
print("torch", torch.__version__, torch.version.cuda)
print("flash_attn", flash_attn.__version__)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
```

已通过的训练 smoke 判定行：

```text
Using FlashAttention-2 for faster training and inference.
Fine-tuning method: LoRA
train_loss = 2.691
```

注意：单卡 Qwen3-8B full SFT 即使 96G 显存也会因 optimizer / 激活 / 参数状态 OOM。sm120 机器上的训练 smoke 应使用 LoRA，不用 full。

## 推理环境：vLLM + FlashInfer + CUDA 13.0

推理性能验证不要复用训练环境。创建单独 conda 环境：

已固化脚本：

```bash
cd /root/autodl-tmp/DB-Opt-R1

# 只安装并验证包，不加载模型
bash scripts/setup_sm120_vllm_flashinfer_env.sh

# 重建环境，并跑 Qwen3-8B vLLM + FlashInfer attention smoke
RECREATE=true RUN_SMOKE=true bash scripts/setup_sm120_vllm_flashinfer_env.sh
```

脚本路径：

```text
scripts/setup_sm120_vllm_flashinfer_env.sh
```

以下命令是脚本展开后的手动版，保留用于排障。

```bash
source ~/miniconda3/etc/profile.d/conda.sh

conda create -p /root/autodl-tmp/conda_envs/dbopt-vllm-flashinfer-cu130 python=3.12 -y
conda activate /root/autodl-tmp/conda_envs/dbopt-vllm-flashinfer-cu130

python -m pip install -U pip setuptools wheel
```

安装 vLLM CUDA 13.0 wheel：

```bash
python -m pip install --no-cache-dir \
  "vllm==0.20.0" \
  --index-url https://wheels.vllm.ai/cu130 \
  --extra-index-url https://pypi.org/simple

python -m pip install --no-cache-dir \
  "transformers==4.57.6" \
  -i https://pypi.org/simple
```

安装 FlashInfer 三件套。不要使用未锁版本的 `flashinfer-python[cu13]`，它可能解析到 `0.6.8.post1`，然后和 `flashinfer-jit-cache==0.6.9+cu130` 错配。

推荐命令：

```bash
python -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache

python -m pip install --no-cache-dir --no-deps --force-reinstall \
  "flashinfer-python==0.6.9" \
  "flashinfer-cubin==0.6.9" \
  -i https://pypi.org/simple

python -m pip install --no-cache-dir --no-deps --force-reinstall \
  "flashinfer-jit-cache==0.6.9+cu130" \
  --index-url https://flashinfer.ai/whl/cu130
```

如果 PyPI 镜像缺 `0.6.9`，改用 direct wheel，同样必须三件套同版本：

```bash
cd /tmp
python -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache

curl -L --retry 5 --retry-delay 3 -o flashinfer_python-0.6.9-py3-none-any.whl \
  "https://sourceforge.net/projects/flashinfer.mirror/files/v0.6.9/flashinfer_python-0.6.9-py3-none-any.whl/download"

curl -L --retry 5 --retry-delay 3 -o flashinfer_cubin-0.6.9-py3-none-any.whl \
  "https://sourceforge.net/projects/flashinfer.mirror/files/v0.6.9/flashinfer_cubin-0.6.9-py3-none-any.whl/download"

curl -L --retry 5 --retry-delay 3 -o flashinfer_jit_cache-0.6.9+cu130-cp39-abi3-manylinux_2_28_x86_64.whl \
  "https://sourceforge.net/projects/flashinfer.mirror/files/v0.6.9/flashinfer_jit_cache-0.6.9%2Bcu130-cp39-abi3-manylinux_2_28_x86_64.whl/download"

python -m pip install --no-cache-dir --no-deps --force-reinstall \
  ./flashinfer_python-0.6.9-py3-none-any.whl \
  ./flashinfer_cubin-0.6.9-py3-none-any.whl \
  ./flashinfer_jit_cache-0.6.9+cu130-cp39-abi3-manylinux_2_28_x86_64.whl
```

## FlashInfer 验证

`sm_120` 必须显式指定 FlashInfer 架构，否则 `flashinfer show-config` 可能自动探测失败：

```bash
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export TORCH_CUDA_ARCH_LIST="12.0"
```

版本与导入检查：

```bash
python - <<'PY'
import torch
from importlib.metadata import version

pkgs = {
    "flashinfer-python": version("flashinfer-python"),
    "flashinfer-cubin": version("flashinfer-cubin"),
    "flashinfer-jit-cache": version("flashinfer-jit-cache"),
}

print("torch =", torch.__version__, torch.version.cuda)
print("gpu =", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print(pkgs)

base = pkgs["flashinfer-jit-cache"].split("+", 1)[0]
assert pkgs["flashinfer-python"] == base, pkgs
assert pkgs["flashinfer-cubin"] == base, pkgs
assert pkgs["flashinfer-jit-cache"].endswith("+cu130"), pkgs

import flashinfer
print("flashinfer import ok:", getattr(flashinfer, "__version__", "unknown"))
PY

flashinfer show-config
```

通过时应看到：

```text
Registered 308 modules
compiled: 308
Not compiled: 0
```

如果看到以下报错，说明当前环境仍不是可用的 FlashInfer sm120 环境：

```text
SM 12.x requires CUDA >= 12.9
```

此时优先检查 `torch.version.cuda` 是否仍是 `12.8`，或者是否遗漏了 `FLASHINFER_CUDA_ARCH_LIST=12.0f`。

## vLLM + FlashInfer attention smoke

不要用 heredoc 直接跑 vLLM，因为 `spawn` 子进程会尝试重新加载主脚本，`<stdin>` 没有真实路径。

创建真实 smoke 脚本：

```bash
cd /root/autodl-tmp/DB-Opt-R1
mkdir -p logs tmp

cat > tmp/sm120_vllm_offline_flashinfer_smoke.py <<'PY'
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def main():
    model = "/root/autodl-tmp/models/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "/no_think\n一句话说明 shared_buffers 是什么。"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    llm = LLM(
        model=model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.70,
        trust_remote_code=True,
        attention_config=AttentionConfig(
            backend=AttentionBackendEnum.FLASHINFER,
        ),
    )

    outs = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=128))
    print("=== OUTPUT ===")
    print(outs[0].outputs[0].text)


if __name__ == "__main__":
    main()
PY
```

运行：

```bash
LOG=logs/sm120_vllm_flashinfer_attention_config_$(date +%Y%m%d_%H%M%S).log

CUDA_VISIBLE_DEVICES=0 \
FLASHINFER_CUDA_ARCH_LIST=12.0f \
TORCH_CUDA_ARCH_LIST="12.0" \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_LOGGING_LEVEL=INFO \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python tmp/sm120_vllm_offline_flashinfer_smoke.py 2>&1 | tee "$LOG"

grep -aEi "Using.*backend|attention backend|attention_config|FLASHINFER|FLASH_ATTN|Traceback|Error|unsupported|OUTPUT" "$LOG" | tail -260
```

通过判定行：

```text
attention_config: AttentionConfig(backend=<AttentionBackendEnum.FLASHINFER ...>)
Using AttentionBackendEnum.FLASHINFER backend.
Warming up FlashInfer attention.
=== OUTPUT ===
```

不要使用：

```bash
VLLM_ATTENTION_BACKEND=FLASHINFER
```

vLLM 0.20.0 会提示：

```text
Unknown vLLM environment variable detected: VLLM_ATTENTION_BACKEND
```

并继续选择默认 backend，例如：

```text
Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', ...]
```

这不算 FlashInfer attention smoke 通过。

## 已踩坑汇总

| 问题 | 现象 | 根因 | 修复 |
|------|------|------|------|
| vLLM import 找 `libcudart.so.13` | `ImportError: libcudart.so.13` | vLLM 扩展和 torch CUDA runtime 混装 | 同一环境内统一 cu130，或 cu128 只作非 FlashInfer 兜底 |
| tokenizer 属性缺失 | `Qwen2Tokenizer has no attribute all_special_tokens_extended` | vLLM 0.11 与 Transformers v5 不兼容 | pin `transformers==4.57.6` |
| FlashInfer 版本错配 | `flashinfer-jit-cache version ... does not match flashinfer version ...` | `flashinfer-python` 与 `jit-cache` 来自不同 release | 三件套锁到同一 release |
| FlashInfer 自动识别 sm120 失败 | `SM 12.x requires CUDA >= 12.9` / `No supported CUDA architectures` | cu128 或未显式指定 arch | 使用 cu130，并设置 `FLASHINFER_CUDA_ARCH_LIST=12.0f` |
| vLLM spawn 找 `<stdin>` | `FileNotFoundError: .../<stdin>` | heredoc 脚本没有真实路径 | 写入临时 `.py` 文件再运行 |
| env var 强制 backend 无效 | `Unknown vLLM environment variable ...` | vLLM 0.20 不认 `VLLM_ATTENTION_BACKEND` | 使用 `AttentionConfig(backend=AttentionBackendEnum.FLASHINFER)` |

## 排障过程与坑位细节

这一节记录实际踩坑过程。后续写自动化环境脚本时，不要只复制最终安装命令，也要保留这些检查点。

### 1. 训练侧 FlashAttention-2

最开始 LLaMA-Factory `FLASH_ATTN=fa2` 并不等于真的使用 FA2。如果 `flash-attn` 没装，日志会出现：

```text
FlashAttention-2 is not installed.
Using torch SDPA for faster training and inference.
```

这只能说明训练可以 fallback 到 SDPA，不能算 FA2 smoke 通过。真正通过必须看到：

```text
Using FlashAttention-2 for faster training and inference.
```

sm120 上 `flash-attn==2.8.3` 的预编译 wheel 可以 import，并且 LLaMA-Factory 能实际走 FA2。不要默认源码编译，尤其磁盘只有 50G 时，源码编译风险和成本都太高。

### 2. 单卡 full SFT OOM

96G 显存不代表 Qwen3-8B full SFT 单卡可行。实际 full smoke 报过：

```text
torch.OutOfMemoryError: CUDA out of memory
GPU 0 has a total capacity of 94.97 GiB
```

原因是 full SFT 不是只放模型权重，还要放 optimizer state、梯度、激活、临时 buffer 等。sm120 上训练 smoke 的目标是验证环境和 kernel，不是验证 full 单卡训练能力，所以应使用 LoRA 1 step。

### 3. vLLM 不要装进训练环境

训练环境已经验证过：

```text
torch 2.8.0+cu128
transformers 4.52.4
deepspeed 0.16.9
llamafactory 0.9.3
flash-attn 2.8.3
```

vLLM 会牵动 `torch`、`transformers`、`triton`、`nvidia-*` runtime 包。直接在训练环境里装 vLLM，很容易把刚跑通的 LLaMA-Factory 环境污染掉。因此 vLLM/FlashInfer 必须放到独立 conda env。

### 4. vLLM 默认 pip 容易混 CUDA runtime

第一次直接装 vLLM 后出现过：

```text
torch = 2.8.0+cu128
ImportError: libcudart.so.13: cannot open shared object file
```

这说明 `vllm._C` 实际链接到了 CUDA 13 runtime，但环境里的 torch 是 cu128。后续脚本不要使用裸的：

```bash
python -m pip install vllm
```

如果目标是 FlashInfer/sm120 性能环境，应显式使用 cu130 wheel 源：

```bash
python -m pip install --no-cache-dir \
  vllm \
  --index-url https://wheels.vllm.ai/cu130 \
  --extra-index-url https://pypi.org/simple
```

### 5. vLLM 0.11 只是 cu128 兜底，不是目标性能环境

`vllm==0.11.0 + torch==2.8.0+cu128` 可以做到基础推理 smoke，但不是 FlashInfer 目标环境。它的价值只是验证模型、tokenizer、chat template、vLLM 基本链路能跑。

这个环境还遇到过 tokenizer 兼容问题：

```text
AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
```

原因是 vLLM 0.11 的 tokenizer cache 仍访问 `all_special_tokens_extended`，而较新的 Transformers API 已变化。修复方式是 pin：

```bash
python -m pip install --no-cache-dir "transformers==4.57.6"
```

不要 patch `site-packages/vllm/transformers_utils/tokenizer.py`。正确做法是 pin 兼容依赖。

### 6. heredoc 与 vLLM spawn 冲突

用 heredoc 运行 vLLM：

```bash
python - <<'PY'
...
PY
```

在 `VLLM_WORKER_MULTIPROC_METHOD=spawn` 下会导致子进程重新加载主脚本时找不到真实文件：

```text
FileNotFoundError: [Errno 2] No such file or directory: '/root/autodl-tmp/DB-Opt-R1/<stdin>'
```

smoke 脚本必须写成真实 `.py` 文件，再执行：

```bash
python tmp/sm120_vllm_offline_flashinfer_smoke.py
```

### 7. FlashInfer 不能在 sm120/cu128 上验证

cu128 下 FlashInfer 报过：

```text
SM 12.x requires CUDA >= 12.9.
No supported CUDA architectures found
```

这不是简单的包没装好，而是 sm120 的 FlashInfer 路线需要 CUDA 12.9+。因此：

| 目标 | 结论 |
|------|------|
| cu128 + vLLM 默认 backend | 可作为功能兜底 |
| cu128 + FlashInfer attention | 不作为目标 |
| cu130 + FlashInfer attention | 目标路线 |

### 8. FlashInfer 包不能混装

FlashInfer 实际由三件套组成：

```text
flashinfer-python
flashinfer-cubin
flashinfer-jit-cache
```

实际遇到过两类错配：

```text
flashinfer-cubin version (0.6.8.post1) does not match flashinfer version (0.6.9)
flashinfer-jit-cache version (0.6.9+cu130) does not match flashinfer version (0.6.8.post1)
```

根因是 `flashinfer-python[cu13]` 或普通 PyPI 解析可能拿到 `0.6.8.post1`，而 `https://flashinfer.ai/whl/cu130` 上的 `flashinfer-jit-cache` 是 `0.6.9+cu130`。脚本必须锁死三件套：

```text
flashinfer-python==0.6.9
flashinfer-cubin==0.6.9
flashinfer-jit-cache==0.6.9+cu130
```

并且安装时尽量使用 `--no-deps`，避免 pip 顺手拉入不期望的 CUDA 依赖。

### 9. pip 可能拉入 CUDA 13 依赖包

错误安装 FlashInfer 时，pip 曾经下载：

```text
cuda_toolkit-13.0.2
nvidia_cudnn_cu13
nvidia_cusparselt_cu13
```

这在 cu128 兜底环境里会造成污染。遇到这种情况应立即停止安装，并清理：

```bash
python -m pip uninstall -y \
  cuda-toolkit nvidia-cudnn-cu13 nvidia-cusparselt-cu13 nvidia-cuda-runtime-cu13 \
  nvidia-cublas-cu13 nvidia-cusolver-cu13 nvidia-cusparse-cu13 nvidia-nccl-cu13 \
  nvidia-nvjitlink-cu13 nvidia-nvshmem-cu13
```

目标 cu130 环境可以接受 CUDA 13 runtime，但仍应由 vLLM cu130 wheel 源统一解析，不要让 FlashInfer 安装步骤随意改 torch/vLLM。

### 10. FlashInfer sm120 需要显式 arch

即使版本组合已经是：

```text
torch 2.11.0+cu130
vllm 0.20.0
flashinfer-python 0.6.9
flashinfer-cubin 0.6.9
flashinfer-jit-cache 0.6.9+cu130
```

`flashinfer show-config` 仍可能自动探测失败：

```text
No modules found. Registering default modules...
Failed to get device capability: SM 12.x requires CUDA >= 12.9.
Module registration failed: No supported CUDA architectures found
```

实际修复是显式指定：

```bash
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export TORCH_CUDA_ARCH_LIST="12.0"
```

之后应看到：

```text
Registered 308 modules
compiled: 308
Not compiled: 0
```

### 11. vLLM 0.20 强制 FlashInfer attention 不能靠环境变量

曾经尝试：

```bash
VLLM_ATTENTION_BACKEND=FLASHINFER
```

vLLM 0.20.0 会明确提示：

```text
Unknown vLLM environment variable detected: VLLM_ATTENTION_BACKEND
```

随后仍选择：

```text
Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', ...]
```

这不算 FlashInfer attention 通过。即使日志里出现：

```text
flashinfer.jit: [Autotuner]: Autotuning process starts
```

也只能说明 FlashInfer JIT/autotune 路径被用到，不代表 attention backend 是 FlashInfer。

真正强制 FlashInfer attention 的方式是 Python API：

```python
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

llm = LLM(
    model=model,
    dtype="bfloat16",
    tensor_parallel_size=1,
    max_model_len=4096,
    gpu_memory_utilization=0.70,
    trust_remote_code=True,
    attention_config=AttentionConfig(
        backend=AttentionBackendEnum.FLASHINFER,
    ),
)
```

真正通过的判定行是：

```text
attention_config: AttentionConfig(backend=<AttentionBackendEnum.FLASHINFER ...>)
Using AttentionBackendEnum.FLASHINFER backend.
Warming up FlashInfer attention.
=== OUTPUT ===
```

## 最终可复现版本

推理性能环境最终通过组合：

```text
conda env: /root/autodl-tmp/conda_envs/dbopt-vllm-flashinfer-cu130
python: 3.12
torch: 2.11.0+cu130
torch CUDA: 13.0
vllm: 0.20.0
transformers: 4.57.6
flashinfer-python: 0.6.9
flashinfer-cubin: 0.6.9
flashinfer-jit-cache: 0.6.9+cu130
FLASHINFER_CUDA_ARCH_LIST: 12.0f
```

最终 vLLM + FlashInfer attention smoke 输出：

```text
Using AttentionBackendEnum.FLASHINFER backend.
Warming up FlashInfer attention.
Processed prompts: 1/1
=== OUTPUT ===
`shared_buffers` 是 PostgreSQL 中用于缓存数据库数据的共享内存区域，提升查询性能。
```
