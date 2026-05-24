"""
在验证集上计算 PPL，并可视化训练曲线

教学重点：
  1. PPL 的计算细节：必须用 token 数做分母，而非 batch 数
  2. PPL 与 loss 的关系：PPL = exp(loss)，两者等价但量纲不同
  3. 训练曲线解读：PPL 何时收敛、是否过拟合

使用方式：
  python evaluate.py                                         # 使用最优模型
  python evaluate.py --checkpoint outputs/checkpoints/epoch1_ppl120.0.pt
  python evaluate.py --plot                                  # 同时绘制训练曲线

依赖：
  pip install torch matplotlib
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import build_model
from train import TokenDataset

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_PATH = OUTPUT_DIR / "training_log.jsonl"


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    vocab_size = ckpt["vocab_size"]
    seq_len = ckpt["seq_len"]
    model = build_model(vocab_size=vocab_size, seq_len=seq_len).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(
        f"加载模型：{ckpt_path.name}（epoch={ckpt.get('epoch', '?')}，checkpoint val PPL={ckpt.get('val_ppl', '?'):.1f}）")
    return model, vocab_size, seq_len


def evaluate_ppl(ckpt_path: Path, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab_size, seq_len = load_model(ckpt_path, device)

    val_path = DATA_DIR / f"val_seq{seq_len}.pt"
    val_ds = TokenDataset(val_path)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for input_ids, targets in val_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            logits = model(input_ids)
            B, T, V = logits.shape
            loss = loss_fn(logits.view(B * T, V), targets.view(B * T))
            total_loss += loss.item()
            total_tokens += B * T

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    logger.info(f"\n{'=' * 50}")
    logger.info(f"验证集评估结果")
    logger.info(f"  总 token 数：{total_tokens:,}")
    logger.info(f"  平均 Cross-Entropy Loss：{avg_loss:.4f}")
    logger.info(f"  PPL（困惑度）= exp({avg_loss:.4f}) = {ppl:.1f}")
    logger.info(f"{'=' * 50}")
    return ppl


def plot_training_curve():
    """绘制训练过程中 PPL 的变化曲线"""
    if not LOG_PATH.exists():
        logger.warning(f"未找到训练日志：{LOG_PATH}")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安装，跳过绘图")
        return

    epochs, train_ppls, val_ppls = [], [], []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            epochs.append(entry["epoch"])
            train_ppls.append(entry["train_ppl"])
            val_ppls.append(entry["val_ppl"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # PPL 曲线
    axes[0].plot(epochs, train_ppls, "b-o", label="Train PPL")
    axes[0].plot(epochs, val_ppls, "r-o", label="Val PPL")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("PPL")
    axes[0].set_title("Perplexity (PPL) Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss 曲线（log scale 下的 PPL = loss）
    train_losses = [math.log(p) for p in train_ppls]
    val_losses = [math.log(p) for p in val_ppls]
    axes[1].plot(epochs, train_losses, "b-o", label="Train Loss")
    axes[1].plot(epochs, val_losses, "r-o", label="Val Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].set_title("Loss Curve (= ln PPL)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "training_curve.png"
    plt.savefig(out_path, dpi=150)
    logger.info(f"训练曲线已保存：{out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint 路径，默认使用 outputs/checkpoints/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--plot", action="store_true", help="是否绘制训练曲线")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else CKPT_DIR / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint：{ckpt_path}，请先运行 train.py")

    evaluate_ppl(ckpt_path, args.batch_size)

    if args.plot:
        plot_training_curve()


if __name__ == "__main__":
    main()
