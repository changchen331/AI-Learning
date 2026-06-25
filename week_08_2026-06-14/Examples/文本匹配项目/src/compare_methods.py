"""
多方法效果对比脚本

对比三种文本匹配方式在 AFQMC validation 集上的效果：
  1. BiEncoder + CosineEmbeddingLoss
  2. BiEncoder + TripletLoss
  3. CrossEncoder + CrossEntropyLoss

教学重点：
  1. 三种方式都用 4 层 BERT，1 epoch 训练，控制变量对比 Loss 函数差异
  2. BiEncoder 需阈值搜索，CrossEncoder 直接 argmax——评估方式本身就是知识点
  3. 速度/精度权衡：CrossEncoder 精度最高但推理慢，BiEncoder 可向量化
  4. 输出对比图：正负样本相似度分布（BiEncoder × 2）+ 混淆矩阵（CrossEncoder）

使用方式：
  python compare_methods.py
  python compare_methods.py --split validation --batch_size 64

依赖：
  pip install torch transformers scikit-learn matplotlib
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset import PairDataset, CrossEncoderDataset
from evaluate import eval_biencoder, eval_crossencoder
from model import build_biencoder, build_crossencoder

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "afqmc"
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
FIG_DIR = ROOT / "outputs" / "figures"
LOG_DIR = ROOT / "outputs" / "logs"

METHODS = [
    {
        "key": "biencoder_cosine",
        "label": "BiEncoder\n(CosineEmbeddingLoss)",
        "ckpt": "biencoder_cosine_best.pt",
        "type": "biencoder",
        "color": "#2196F3",
    },
    {
        "key": "biencoder_triplet",
        "label": "BiEncoder\n(TripletLoss)",
        "ckpt": "biencoder_triplet_best.pt",
        "type": "biencoder",
        "color": "#4CAF50",
    },
    {
        "key": "crossencoder",
        "label": "CrossEncoder\n(CrossEntropyLoss)",
        "ckpt": "crossencoder_best.pt",
        "type": "crossencoder",
        "color": "#FF9800",
    },
]


# ── 加载并评估单个方法 ─────────────────────────────────────────────────────

def load_and_eval(method, tokenizer, device, split, batch_size):
    ckpt_path = CKPT_DIR / method["ckpt"]
    if not ckpt_path.exists():
        print(f"  [SKIP] checkpoint 不存在: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})

    if method["type"] == "biencoder":
        model = build_biencoder(
            bert_path=str(BERT_PATH),
            pool=saved.get("pool", "mean"),
            num_hidden_layers=saved.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        data_path = DATA_DIR / f"{split}.jsonl"
        ds = PairDataset(data_path, tokenizer, max_length=saved.get("max_length", 64))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = eval_biencoder(model, loader, device)

    else:  # crossencoder
        model = build_crossencoder(
            bert_path=str(BERT_PATH),
            num_hidden_layers=saved.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        data_path = DATA_DIR / f"{split}.jsonl"
        ds = CrossEncoderDataset(data_path, tokenizer, max_length=128)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = eval_crossencoder(model, loader, device)

    metrics["model"] = model
    metrics["ckpt"] = ckpt
    return metrics


# ── 对比可视化 ────────────────────────────────────────────────────────────

def plot_comparison_bar(results, save_path):
    """准确率 / F1 对比柱状图"""
    names = [m["label"] for m in results]
    accs = [m["accuracy"] for m in results]
    f1s = [m["f1"] for m in results]
    colors = [m["color"] for m in results]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w / 2, accs, w, label="Accuracy", color=colors, alpha=0.85)
    bars2 = ax.bar(x + w / 2, f1s, w, label="F1 (weighted)", color=colors, alpha=0.5,
                   hatch="//", edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Method Comparison on AFQMC Validation (4-layer BERT, 1 epoch)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


def plot_sim_distributions(biencoder_results, save_path):
    """所有 BiEncoder 方法的相似度分布叠放对比"""
    n = len(biencoder_results)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, biencoder_results):
        sims = np.array(m["similarities"])
        labels = np.array(m["labels"])
        ax.hist(sims[labels == 1], bins=40, alpha=0.6, label="positive", color="#2196F3", density=True)
        ax.hist(sims[labels == 0], bins=40, alpha=0.6, label="negative", color="#F44336", density=True)
        ax.axvline(m["threshold"], color="black", linestyle="--",
                   label=f"threshold={m['threshold']:.2f}")
        ax.set_title(m["label"].replace("\n", " "))
        ax.set_xlabel("Cosine Similarity")
        ax.legend(fontsize=8)

    fig.suptitle("BiEncoder Similarity Distribution (positive vs negative)", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="三种文本匹配方法效果对比")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--batch_size", default=64, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}  评估集: {args.split}")

    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── 逐方法评估 ────────────────────────────────────────────────────────
    all_results = []
    for m in METHODS:
        print(f"\n{'=' * 55}")
        print(f"加载 {m['key']} ...")
        metrics = load_and_eval(m, tokenizer, device, args.split, args.batch_size)
        if metrics is None:
            continue
        metrics.update({"label": m["label"], "color": m["color"], "key": m["key"],
                        "type": m["type"]})
        all_results.append(metrics)

    if not all_results:
        print("没有可用的 checkpoint，请先运行训练脚本。")
        return

    # ── 控制台对比表 ──────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"{'方法':<30} {'Accuracy':>9} {'F1(weighted)':>13} {'额外信息':>15}")
    print(f"{'-' * 65}")
    for m in all_results:
        extra = (f"threshold={m['threshold']:.2f}"
                 if m["type"] == "biencoder" else "argmax")
        print(f"  {m['key']:<28} {m['accuracy']:>9.4f} {m['f1']:>13.4f} {extra:>15}")

    print(f"\n{'─' * 65}")
    print("结论速览：")
    best_acc = max(all_results, key=lambda x: x["accuracy"])
    best_f1 = max(all_results, key=lambda x: x["f1"])
    print(f"  最高 Accuracy : {best_acc['key']} ({best_acc['accuracy']:.4f})")
    print(f"  最高 F1       : {best_f1['key']}  ({best_f1['f1']:.4f})")

    # 列出不同 Loss 的 BiEncoder 对比
    bi_results = [m for m in all_results if m["type"] == "biencoder"]
    if len(bi_results) == 2:
        a, b = bi_results
        delta_acc = b["accuracy"] - a["accuracy"]
        delta_f1 = b["f1"] - a["f1"]
        print(f"\n  Cosine vs Triplet (Δ):")
        print(f"    Accuracy: {delta_acc:+.4f}  F1: {delta_f1:+.4f}")
        if abs(delta_f1) < 0.01:
            print("    → 两种 Loss 差距不大（1 epoch + 少量三元组限制了 Triplet 的优势）")
        elif delta_f1 > 0:
            print("    → TripletLoss 更优，三元组对语义距离的约束更精确")
        else:
            print("    → CosineEmbeddingLoss 更优，AFQMC 数据量下直接对标签优化更稳定")

    # ── 保存对比日志 ──────────────────────────────────────────────────────
    skip_keys = {"model", "similarities", "labels", "logits", "ckpt"}

    def _to_py(v):
        if hasattr(v, "item"):  # numpy scalar / torch scalar
            return v.item()
        return v

    log = [{k: _to_py(v) for k, v in m.items() if k not in skip_keys}
           for m in all_results]
    log_path = LOG_DIR / "method_comparison.json"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"\n对比日志 → {log_path}")

    # ── 可视化 ────────────────────────────────────────────────────────────
    plot_comparison_bar(all_results, FIG_DIR / "method_comparison_bar.png")

    bi_with_sim = [m for m in all_results if m["type"] == "biencoder" and "similarities" in m]
    if bi_with_sim:
        plot_sim_distributions(bi_with_sim, FIG_DIR / "biencoder_sim_distributions.png")


if __name__ == "__main__":
    main()
