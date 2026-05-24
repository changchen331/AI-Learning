"""
将原始文本 tokenize，拼接成固定长度块，存为 PyTorch 张量文件

教学重点：
  1. 预训练数据处理：文本 → token id → 连续长序列 → 切块
  2. "拼接后切块" vs "按句截断"：预训练用前者，充分利用每个 token
  3. 训练/验证集划分：按 token 数量划分，而非按文章数量
  4. 数据规模估算：1亿 token / seq_len=256 = ~390k 个训练样本

使用方式：
  python prepare_dataset.py                        # 默认参数
  python prepare_dataset.py --seq_len 512          # 更长上下文
  python prepare_dataset.py --max_tokens 20000000  # 快速验证（2000万token）

依赖：
  pip install transformers
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from transformers.models.bert.tokenization_bert import BertTokenizerFast

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PRETRAIN_MODEL_DIR = BASE_DIR.parent.parent / "项目材料准备" / "pretrain_models" / "bert-base-chinese"

# bert-base-chinese tokenizer 的特殊 token
SEP_TOKEN_ID = 102  # [SEP] 用于文章间分隔


def get_tokenizer():
    """加载 bert-base-chinese tokenizer，优先用本地路径"""
    local_path = Path(__file__).parent.parent.parent / "pretrain_models" / "bert-base-chinese"
    if local_path.exists():
        logger.info(f"使用本地 tokenizer：{local_path}")
        return BertTokenizerFast.from_pretrained(str(local_path))
    logger.info("从 HuggingFace 下载 bert-base-chinese tokenizer...")
    return BertTokenizerFast.from_pretrained("bert-base-chinese")


def build_dataset(seq_len: int = 256, val_ratio: float = 0.05, max_tokens: int = None):
    jsonl_path = DATA_DIR / "wiki_zh.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"未找到数据文件 {jsonl_path}，请先运行 download_data.py")

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    logger.info(f"Tokenizer vocab size: {vocab_size}")

    # ── 第一步：读取文章，tokenize，拼接成一条长 token 流 ──────────────────────
    # 核心思路：把所有文章首尾相接，文章之间插入 [SEP] 作为边界标记
    # 好处：没有任何 token 被浪费，每个训练样本都是完整的 seq_len 长度
    logger.info("开始 tokenize 并拼接所有文章...")
    all_ids = []
    n_articles = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            # 只对正文 tokenize，不加 [CLS]/[SEP]，让模型自己学文章结构
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            all_ids.append(SEP_TOKEN_ID)  # 文章边界标记
            n_articles += 1

            if max_tokens and len(all_ids) >= max_tokens:
                break

            if n_articles % 5000 == 0:
                logger.info(f"  已处理 {n_articles} 篇，当前 token 数：{len(all_ids):,}")

    total_tokens = len(all_ids)
    logger.info(f"总 token 数：{total_tokens:,}（来自 {n_articles} 篇文章）")

    # ── 第二步：切成固定长度的块 ──────────────────────────────────────────────
    # 每个块是 seq_len+1 个 token：input=前seq_len个，target=后seq_len个（右移1位）
    n_chunks = (total_tokens - 1) // seq_len
    logger.info(f"切块数：{n_chunks}（seq_len={seq_len}）")

    ids_tensor = torch.tensor(all_ids[:n_chunks * seq_len + 1], dtype=torch.long)

    # ── 第三步：划分训练集和验证集 ────────────────────────────────────────────
    val_size = max(1, int(n_chunks * val_ratio))
    train_size = n_chunks - val_size
    logger.info(f"训练样本：{train_size:,}，验证样本：{val_size:,}")

    # 存储为 (N, seq_len+1) 的张量，每行 = 一个训练样本（含 target 的右移）
    train_data = ids_tensor[: train_size * seq_len + 1].unfold(0, seq_len + 1, seq_len)[:-1]
    val_data = ids_tensor[train_size * seq_len: (train_size + val_size) * seq_len + 1].unfold(0, seq_len + 1, seq_len)[
        :-1]

    train_path = DATA_DIR / f"train_seq{seq_len}.pt"
    val_path = DATA_DIR / f"val_seq{seq_len}.pt"

    torch.save({"data": train_data, "vocab_size": vocab_size, "seq_len": seq_len}, train_path)
    torch.save({"data": val_data, "vocab_size": vocab_size, "seq_len": seq_len}, val_path)

    logger.info(f"训练集已保存：{train_path}（{train_data.shape}）")
    logger.info(f"验证集已保存：{val_path}（{val_data.shape}）")
    return train_path, val_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=256, help="序列长度，默认 256")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证集比例，默认 0.05")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="限制最大 token 数，如 20000000 表示 2000万（快速验证用）")
    args = parser.parse_args()
    build_dataset(args.seq_len, args.val_ratio, args.max_tokens)


if __name__ == "__main__":
    main()
