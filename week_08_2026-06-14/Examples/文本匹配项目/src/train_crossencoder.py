"""
CrossEncoder 训练脚本（交互型文本匹配）

教学重点：
  1. CrossEncoder vs BiEncoder 的关键差异：
       CrossEncoder 让两句在 BERT 每一层都充分交互（Self-Attention 跨句），
       表达能力更强但无法预计算，不适合向量检索；
       BiEncoder 两句独立编码，可预计算向量，适合大规模检索（如 RAG Recall）
  2. 输入格式：[CLS] s1 [SEP] s2 [SEP]，token_type_ids 区分两段
       这是 BERT 原始预训练任务（NSP，Next Sentence Prediction）的格式
  3. CrossEncoder 评估与普通分类完全相同，无需阈值搜索（argmax 即为预测）

使用方式：
  # 默认参数（4 层 BERT，3 epoch）
  python train_crossencoder.py

  # 自定义参数
  python train_crossencoder.py --num_hidden_layers 6 --epochs 5 --batch_size 16

依赖：
  pip install torch transformers scikit-learn tqdm
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from dataset import build_crossencoder_loaders
from evaluate import eval_crossencoder
from model import build_crossencoder

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "afqmc"
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"


# ── 训练一个 epoch ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, criterion,
                    device, epoch, total_epochs, grad_accum):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [CrossEncoder]", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)

        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        pbar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    return total_loss / total_samples, total_correct / total_samples


# ── 主训练流程 ────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"BERT 层数: {args.num_hidden_layers}  Epochs: {args.epochs}  "
          f"Batch size: {args.batch_size}")

    # ── Tokenizer & DataLoader ────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    print("\nDataLoader 构建中...")
    train_loader, val_loader, _ = build_crossencoder_loaders(
        args.data_dir, tokenizer,
        max_length=args.max_length, batch_size=args.batch_size,
    )

    # ── 模型 ─────────────────────────────────────────────────────────────
    print("\n构建模型...")
    model = build_crossencoder(
        bert_path=args.bert_path,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)

    # ── 分层学习率 ────────────────────────────────────────────────────────
    bert_params = list(model.bert.parameters())
    head_params = (list(model.dropout.parameters()) +
                   list(model.classifier.parameters()))

    optimizer = AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"总训练步数: {total_steps}  Warmup 步数: {warmup_steps}")

    # AFQMC 正负比约 31%:69%，CrossEntropyLoss 默认不加权
    # 如需加权，可参考 bert_text_classification 的 compute_class_weight 做法
    criterion = nn.CrossEntropyLoss()

    # ── 训练循环 ──────────────────────────────────────────────────────────
    ckpt_path = CKPT_DIR / "crossencoder_best.pt"
    best_val_f1 = 0.0
    log_records = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, args.epochs, args.grad_accum,
        )

        val_metrics = eval_crossencoder(model, val_loader, device)
        elapsed = time.time() - t0

        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["f1"]
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
              f"{elapsed:.0f}s")

        log_records.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_acc": val_acc, "val_f1": val_f1, "elapsed_s": elapsed,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✓ 新最优模型已保存 → {ckpt_path}  (val_f1={val_f1:.4f})")

    # ── 训练完成，保存日志 ────────────────────────────────────────────────
    log_path = LOG_DIR / "crossencoder_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)
    print(f"\n训练完成。最优 val_f1={best_val_f1:.4f}")
    print(f"训练日志 → {log_path}")
    print(f"最优 checkpoint → {ckpt_path}")
    print(f"\n运行评估：python evaluate.py --model_type crossencoder --ckpt {ckpt_path}")


# ── 参数解析 ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CrossEncoder 训练（交互型文本匹配）")
    parser.add_argument("--bert_path", default=str(BERT_PATH), type=str)
    parser.add_argument("--data_dir", default=str(DATA_DIR), type=str)
    parser.add_argument("--num_hidden_layers", default=4, type=int,
                        help="BERT Transformer 层数（默认 4 层；全量 12 层留给学生自行实验）")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_length", default=128, type=int,
                        help="句对总最大 token 数（两句拼接，建议 128）")
    parser.add_argument("--lr", default=2e-5, type=float, help="BERT 层学习率")
    parser.add_argument("--head_lr_mult", default=5.0, type=float, help="分类头学习率倍数")
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--grad_accum", default=1, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
