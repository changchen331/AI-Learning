#!/bin/bash
# 启动 vLLM OpenAI 兼容 server
#
# 教学重点：
#   1. 一条命令把 HuggingFace 模型变成 OpenAI 兼容 API
#   2. 关键启动参数：max-model-len / gpu-memory-utilization / dtype
#   3. 启动后访问 http://localhost:8000/v1/chat/completions 即可调用
#
# 使用方式（在 WSL2 Ubuntu 内执行）：
#   cd /mnt/d/badou/项目材料准备/vllm_deployment_demo/src/
#   bash start_server.sh
#
# 环境依赖：
#   已激活 ~/vllm_env（vLLM 0.9.2 + torch 2.7+cu126）

set -e

# ── 配置 ─────────────────────────────────────────────────────
MODEL_PATH="/mnt/d/badou/项目材料准备/pretrain_models/Qwen2-0.5B-Instruct"
SERVED_NAME="qwen2-0.5b"    # 客户端 API 里使用的模型名（与实际路径解耦）
PORT=8000
MAX_MODEL_LEN=2048          # 最大上下文长度（0.5B 模型不需要太长）
GPU_MEM_UTIL=0.6            # 占用 60% 显存（给其他程序留余地）
DTYPE="float16"             # Qwen2 半精度足够，bf16 也可

# ── 激活 venv ────────────────────────────────────────────────
if [ -z "$VIRTUAL_ENV" ]; then
    source ~/vllm_env/bin/activate
fi

# ── 防止 WSL 下 torch/numpy OpenMP 冲突 ─────────────────────
export KMP_DUPLICATE_LIB_OK=TRUE

echo "============================================"
echo "  启动 vLLM OpenAI Server"
echo "  模型路径: $MODEL_PATH"
echo "  对外名称: $SERVED_NAME"
echo "  端口:     $PORT"
echo "  max_len:  $MAX_MODEL_LEN"
echo "  显存占用: ${GPU_MEM_UTIL} (约 5GB / 8GB)"
echo "============================================"
echo ""
echo "启动后用以下命令测试："
echo "  curl http://localhost:${PORT}/v1/models"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype "$DTYPE" \
    --enforce-eager \
    --host 0.0.0.0
