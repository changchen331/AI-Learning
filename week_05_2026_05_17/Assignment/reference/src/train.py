"""
GPT 预训练主循环

教学重点：
  1. 自回归语言模型的训练目标：预测下一个 token，交叉熵 loss
     input  = tokens[0..T-1]，target = tokens[1..T]（右移一位）
  2. PPL（困惑度）= exp(avg_cross_entropy_loss)
     PPL 越低 → 模型对验证集越"不困惑" → 语言建模能力越强
  3. AdamW + 余弦学习率调度：预训练标准配置
  4. Gradient Clipping：防止梯度爆炸，预训练必备
  5. Checkpoint 保存策略：每 epoch 保存，同时记录最优 val PPL

使用方式：
  python train.py                        # 默认配置
  python train.py --epochs 5             # 训练轮数
  python train.py --batch_size 16        # 显存不足时减小
  python train.py --lr 3e-4              # 学习率

依赖：
  pip install torch transformers
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from model import build_model

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_PATH = OUTPUT_DIR / "training_log.jsonl"


class TokenDataset(Dataset):
    """
    从 .pt 文件加载 token 数据集

    每个样本是 (seq_len+1,) 的 token 序列：
      input  = sample[:-1]  即 tokens[0..T-1]
      target = sample[1:]   即 tokens[1..T]
    """

    def __init__(self, pt_path: Path):
        ckpt = torch.load(pt_path, weights_only=True)
        self.data = ckpt["data"]  # (N, seq_len+1)
        self.vocab_size = ckpt["vocab_size"]
        self.seq_len = ckpt["seq_len"]
        logger.info(f"加载数据集：{pt_path.name}，共 {len(self.data):,} 个样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]  # (seq_len+1,)
        return chunk[:-1], chunk[1:]  # input, target


def compute_ppl(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    计算验证集 PPL

    PPL = exp(H)，其中 H = -1/N * Σ log P(token_i | context)
    等价于 PPL = exp(avg_cross_entropy_loss)

    PPL 的直觉：模型平均在 PPL 个 token 中"选择"下一个词
    好的语言模型：PPL < 100；随机猜测：PPL ≈ vocab_size (21128)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for input_ids, targets in loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            logits = model(input_ids)  # (B, T, V)
            B, T, V = logits.shape
            loss = loss_fn(logits.view(B * T, V), targets.view(B * T))
            total_loss += loss.item()
            total_tokens += B * T

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    model.train()
    return ppl


def train(
        epochs: int = 3,
        batch_size: int = 32,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        seq_len: int = 256,
        num_workers: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备：{device}")

    # ── 数据加载 ──────────────────────────────────────────────────────────────
    train_path = DATA_DIR / f"train_seq{seq_len}.pt"
    val_path = DATA_DIR / f"val_seq{seq_len}.pt"
    if not train_path.exists():
        raise FileNotFoundError(f"未找到 {train_path}，请先运行 prepare_dataset.py")

    train_ds = TokenDataset(train_path)
    val_ds = TokenDataset(val_path)
    vocab_size = train_ds.vocab_size

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(device.type == "cuda"))

    # ── 模型 ──────────────────────────────────────────────────────────────────
    model = build_model(vocab_size=vocab_size, seq_len=seq_len).to(device)
    logger.info(f"模型参数量：{model.count_parameters() / 1e6:.1f}M")

    # ── 优化器：AdamW + 余弦 LR ───────────────────────────────────────────────
    # weight_decay 只施加在权重矩阵（2D+）上，bias 和 LayerNorm 不衰减
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.95))

    total_steps = len(train_loader) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.1)
    loss_fn = nn.CrossEntropyLoss()

    # ── 初始基线 PPL ──────────────────────────────────────────────────────────
    logger.info("计算训练前基线 PPL（随机初始化）...")
    baseline_ppl = compute_ppl(model, val_loader, device)
    logger.info(f"基线 val PPL：{baseline_ppl:.1f}（随机猜测约等于 vocab_size={vocab_size}）")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_ppl = float("inf")
    global_step = 0

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for step, (input_ids, targets) in enumerate(train_loader, 1):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            logits = model(input_ids)  # (B, T, V)
            B, T, V = logits.shape

            # 自回归损失：每个位置预测下一个 token
            loss = loss_fn(logits.view(B * T, V), targets.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping：将梯度的 L2 范数截断到 grad_clip
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if step % 200 == 0:
                avg_loss = epoch_loss / n_batches
                ppl = math.exp(avg_loss)
                cur_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch}/{epochs} Step {step}/{len(train_loader)} | "
                    f"loss={avg_loss:.4f} PPL={ppl:.1f} lr={cur_lr:.2e} grad_norm={grad_norm:.3f}"
                )

        # ── Epoch 结束：计算验证集 PPL ─────────────────────────────────────────
        val_ppl = compute_ppl(model, val_loader, device)
        train_ppl = math.exp(epoch_loss / n_batches)
        cur_lr = scheduler.get_last_lr()[0]

        logger.info(
            f"\n{'=' * 60}\n"
            f"Epoch {epoch} 完成 | "
            f"train PPL={train_ppl:.1f} | val PPL={val_ppl:.1f} | lr={cur_lr:.2e}\n"
            f"{'=' * 60}"
        )

        # 记录训练日志
        log_entry = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": epoch_loss / n_batches,
            "train_ppl": train_ppl,
            "val_ppl": val_ppl,
            "lr": cur_lr,
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # 保存每个 epoch 的 checkpoint
        ckpt_path = CKPT_DIR / f"epoch{epoch}_ppl{val_ppl:.1f}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_ppl": val_ppl,
            "vocab_size": vocab_size,
            "seq_len": seq_len,
        }, ckpt_path)
        logger.info(f"Checkpoint 已保存：{ckpt_path.name}")

        # 保存最优模型
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_path = CKPT_DIR / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_ppl": val_ppl,
                "vocab_size": vocab_size,
                "seq_len": seq_len,
            }, best_path)
            logger.info(f"最优模型已更新 → val PPL={best_val_ppl:.1f}")

    logger.info(f"\n训练完成！最优 val PPL = {best_val_ppl:.1f}")
    logger.info(f"训练日志：{LOG_PATH}")
    logger.info(f"最优模型：{CKPT_DIR / 'best_model.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
