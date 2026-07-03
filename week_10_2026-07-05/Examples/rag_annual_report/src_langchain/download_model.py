"""
BGE 模型下载脚本

将 BAAI/bge-small-zh-v1.5 下载到项目的 models/ 目录，
而不是 HuggingFace 默认的用户缓存路径（~/.cache/huggingface/）。

好处：
  - 项目自包含，迁移/分享时模型跟着走
  - 避免和其他项目的模型混在一起
  - 学生清楚地知道模型文件在哪里

模型大小：约 90MB

使用方式：
  cd rag_annual_report
  python src_langchain/download_model.py

下载完成后模型在：
  models/bge-small-zh-v1.5/

如果下载速度慢，可以设置镜像：
  set HF_ENDPOINT=https://hf-mirror.com   (Windows)
  export HF_ENDPOINT=https://hf-mirror.com  (Linux/Mac)
"""

import os
import sys
from pathlib import Path

# 项目根目录
BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_NAME  = "BAAI/bge-small-zh-v1.5"
LOCAL_DIR   = MODELS_DIR / "bge-small-zh-v1.5"


def download():
    print(f"目标目录: {LOCAL_DIR}")

    if LOCAL_DIR.exists() and any(LOCAL_DIR.iterdir()):
        # 检查关键文件是否存在
        key_files = ["config.json", "tokenizer.json"]
        if all((LOCAL_DIR / f).exists() for f in key_files):
            print(f"模型已存在，跳过下载: {LOCAL_DIR}")
            return str(LOCAL_DIR)

    print(f"开始下载 {MODEL_NAME} → {LOCAL_DIR}")
    print("首次下载约 90MB，请耐心等待...\n")

    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=str(LOCAL_DIR),
            local_dir_use_symlinks=False,   # 真实文件，不用软链接
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],  # 只要 PyTorch 格式
        )
        print(f"\n下载完成！模型路径: {path}")
        return path
    except Exception as e:
        print(f"\n下载失败: {e}")
        print("\n可以尝试：")
        print("  1. 设置镜像: set HF_ENDPOINT=https://hf-mirror.com")
        print("  2. 手动从 https://hf-mirror.com/BAAI/bge-small-zh-v1.5 下载")
        print(f"     解压到: {LOCAL_DIR}")
        sys.exit(1)


def verify(model_path: str):
    """验证模型可以正常加载。"""
    print("\n验证模型可用性...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_path)
        test_vec = model.encode(["测试句子"], normalize_embeddings=True)
        print(f"验证通过！embedding 维度: {test_vec.shape[1]}")
    except Exception as e:
        print(f"验证失败: {e}")
        print("模型文件可能不完整，请删除后重新下载")


if __name__ == "__main__":
    path = download()
    verify(path)
