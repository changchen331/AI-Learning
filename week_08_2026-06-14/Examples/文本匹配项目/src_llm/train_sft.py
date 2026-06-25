"""
LLM SFT（监督微调）训练脚本 — 基于 LoRA 高效微调 Qwen2-0.5B-Instruct 做文本匹配

教学重点：
  1. 文本匹配的指令微调格式：输入是两句话，输出是"相似"或"不相似"（2~3 token）
     与 NER 的区别：TARGET 极短，loss 极度集中，训练收敛快
  2. Loss masking：只在"相似"/"不相似"这 1~2 个 token 上计算 loss
  3. LoRA 高效微调（--full_ft 开关可切换全量微调）
  4. 与 BiEncoder/CrossEncoder 对比：判别式 vs 生成式，各自的适用场景

使用方式：
  python train_sft.py                        # LoRA，5000 条快速演示（默认）
  python train_sft.py --num_train -1         # LoRA，全部 34K 条
  python train_sft.py --epochs 1             # 快速验证流程

  # 全量微调（需显存 ≥ 16GB）
  python train_sft.py --full_ft --lr 2e-5

依赖：
  pip install torch transformers peft tqdm   # LoRA 模式
  pip install torch transformers tqdm        # 全量微调模式（不需要 peft）
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from peft import get_peft_model, LoraConfig, TaskType

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "afqmc"
MODEL_PATH = ROOT.parent.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"
OUTPUT_DIR = ROOT / "outputs"

SYSTEM_PROMPT = (
    "你是一个语义匹配助手。判断两句话语义是否相同，"
    "只输出【相似】或【不相似】，不要输出其他内容。"
)

LABEL_MAP = {1: "【相似】", 0: "【不相似】"}


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class SFTDataset(Dataset):
    """
    把文本匹配数据转换为 chat-format SFT 训练样本。

    每条样本结构：
      ┌────────────────────────────────────────────────────────────┐
      │ <system>...<user>句子A：...\n句子B：...\n是否相似：<asst>\n│  → labels = -100
      │ 相似 / 不相似  <EOS>                                       │  → labels = token id
      └────────────────────────────────────────────────────────────┘

    TARGET 只有 2~3 个 token，是所有任务中最短的。
    """

    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label_text = LABEL_MAP[item["label"]]

        # ── Step 1：构建 prompt 文本（tokenize=False，兼容 transformers 5.x）──
        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"句子A：{item['sentence1']}\n句子B：{item['sentence2']}\n是否相似："},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        # ── Step 2：response = 标签文本 + EOS ─────────────────────────────────
        response_ids = (
                self.tokenizer.encode(label_text, add_special_tokens=False)
                + [self.tokenizer.eos_token_id]
        )

        # ── Step 3：拼接 + 截断 + loss mask ──────────────────────────────────
        input_ids = (prompt_ids + response_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch, pad_id):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_list, labels_list, mask_list = [], [], []
    for item in batch:
        n = item["input_ids"].size(0)
        pad = max_len - n
        input_ids_list.append(torch.cat([item["input_ids"],
                                         torch.full((pad,), pad_id, dtype=torch.long)]))
        labels_list.append(torch.cat([item["labels"],
                                      torch.full((pad,), -100, dtype=torch.long)]))
        mask_list.append(torch.cat([torch.ones(n, dtype=torch.long),
                                    torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="LLM SFT 文本匹配训练（LoRA / 全量微调）")
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--num_train", default=5000, type=int,
                        help="训练样本数，-1 使用全部 34K 条")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--grad_accum", default=4, type=int)
    parser.add_argument("--lr", default=None, type=float,
                        help="学习率；默认 LoRA=2e-4，全量=2e-5（自动判断）")
    parser.add_argument("--max_length", default=128, type=int)
    # 全量微调开关
    parser.add_argument("--full_ft", action="store_true",
                        help="全量微调：跳过 LoRA，更新所有 495M 参数（需显存 ≥ 16GB）")
    # LoRA 超参（full_ft 时忽略）
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.lr is None:
        args.lr = 2e-5 if args.full_ft else 2e-4

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / ("sft_full_ckpt" if args.full_ft else "sft_adapter")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_str = "全量微调" if args.full_ft else "LoRA 微调"
    print(f"使用设备: {device}  |  微调模式: {mode_str}")

    # ── 加载数据（JSONL 格式）────────────────────────────────────────────────
    def load_jsonl(path):
        with open(path, encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    train_raw = load_jsonl(data_dir / "train.jsonl")
    val_raw = load_jsonl(data_dir / "validation.jsonl")

    # ── 正负平衡采样（AFQMC 正负比 31:69，不平衡会导致模型全预测负例）──────
    pos = [r for r in train_raw if r["label"] == 1]
    neg = [r for r in train_raw if r["label"] == 0]
    if args.num_train > 0:
        n_each = min(args.num_train // 2, len(pos), len(neg))
    else:
        n_each = min(len(pos), len(neg))  # 全量时取较少类的数量
    train_raw = random.sample(pos, n_each) + random.sample(neg, n_each)
    random.shuffle(train_raw)
    print(f"训练集: {len(train_raw)} 条（正{n_each}+负{n_each}，平衡采样）| 验证集（前500条）: 500 条")

    # ── 加载 Tokenizer ─────────────────────────────────────────────────────────
    print(f"\n加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_path).resolve()), trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 构建数据集 ─────────────────────────────────────────────────────────────
    train_dataset = SFTDataset(train_raw, tokenizer, args.max_length)
    val_dataset = SFTDataset(val_raw[:500], tokenizer, args.max_length)

    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2,
                            shuffle=False, collate_fn=_collate)

    # ── 加载模型 ───────────────────────────────────────────────────────────────
    print(f"加载 base model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_path).resolve()),
        dtype=torch.float32,  # transformers 5.x 用 dtype= 不用 torch_dtype=
        trust_remote_code=True,
    )

    # ── LoRA 或全量微调 ────────────────────────────────────────────────────────
    if args.full_ft:
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {total:,} || all params: {total:,} || trainable%: 100.0000")
    else:
        if not PEFT_AVAILABLE:
            raise ImportError("LoRA 模式需要 peft 库：pip install peft>=0.14.0")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model = model.to(device)

    # ── 优化器 ────────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    print(f"总训练步数: {total_steps}（batch={args.batch_size}, "
          f"grad_accum={args.grad_accum}, epochs={args.epochs}, lr={args.lr}）\n")

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    log_records = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss

            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / max(total_tokens, 1)

        # ── 验证 loss ─────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_tokens = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                n_tokens = (labels != -100).sum().item()
                val_loss += outputs.loss.item() * n_tokens
                val_tokens += n_tokens
        avg_val_loss = val_loss / max(val_tokens, 1)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f} | {elapsed:.0f}s")

        log_records.append({
            "epoch": epoch, "train_loss": avg_train_loss,
            "val_loss": avg_val_loss, "elapsed_s": elapsed,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            ckpt_label = "完整模型" if args.full_ft else "LoRA adapter"
            print(f"  ✓ 最优{ckpt_label}已保存 → {ckpt_dir}  (val_loss={avg_val_loss:.4f})")

    # ── 保存训练日志 ──────────────────────────────────────────────────────────
    log_tag = "full_ft" if args.full_ft else "sft"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{log_tag}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    ckpt_label = "完整模型" if args.full_ft else "LoRA adapter"
    print(f"\n训练完成。最优 val_loss={best_val_loss:.4f}")
    print(f"训练日志 → {log_path}")
    print(f"{ckpt_label} → {ckpt_dir}")
    print(f"\n下一步：python evaluate_sft.py 查看 Accuracy / F1 与多方对比")


if __name__ == "__main__":
    main()
