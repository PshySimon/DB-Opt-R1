"""
Modal 部署 vLLM 推理服务（OpenAI 兼容）

用于评估 Qwen3-4B 等模型的 DB 调优能力。

部署：
    modal deploy scripts/modal_vllm.py

查看 URL：
    modal app list

停止：
    modal app stop vllm-eval

本地评估调用：
    python3 -m evaluate.run \
        --scenarios data_pipeline/data/scenarios/ \
        --knob-space configs/knob_space.yaml \
        --cost-model cost_model/checkpoints/v7_lgbm_dedup \
        --api-key "EMPTY" \
        --api-base "https://<your-workspace>--vllm-eval-serve.modal.run/v1" \
        --model "Qwen/Qwen3-4B" \
        --output eval_results/qwen3_4b/ \
        --max-turns 10 --parallel 5
"""

import modal

# ===== 配置 =====
MODEL_NAME = "Qwen/Qwen3-4B"
GPU_TYPE = modal.gpu.L40S()
VLLM_PORT = 8000

# ===== 镜像 =====
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .uv_pip_install(
        "vllm==0.8.5.post1",
        "transformers>=4.51,<5",
        "huggingface-hub",
        "hf_transfer",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

# ===== 模型权重缓存 =====
model_volume = modal.Volume.from_name("vllm-model-cache", create_if_missing=True)

app = modal.App("vllm-eval")


@app.function(
    image=vllm_image,
    gpu=GPU_TYPE,
    cpu=4.0,
    memory=16384,  # 16 GiB
    volumes={"/root/.cache/huggingface": model_volume},
    max_containers=1,
    scaledown_window=600,  # 空闲 10 分钟自动关，避免频繁冷启动
    timeout=3600,
)
@modal.web_server(port=VLLM_PORT, startup_timeout=300)
def serve():
    import subprocess

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.85",
        "--dtype", "bfloat16",
        "--trust-remote-code",
    ]

    subprocess.Popen(cmd)
