"""
BadCase 分析与优化方向建议

教学重点：
  1. FP（假阳性）vs FN（假阴性）的不同成因：
       FP：模型认为相似但实际不同 → 通常因为表面词汇高度重叠
       FN：模型认为不同但实际相似 → 通常因为换了表达方式
  2. 高置信度错误 vs 临界错误：错在边界和错在"确信"说明不同问题
  3. 从 bad case 出发推导优化方向（数据、模型、训练策略）
  4. 相似度分布中正负样本的重叠区 = 模型最难判断的区间

使用方式：
  # 分析 BiEncoder（CosineEmbeddingLoss）
  python analyze_badcases.py

  # 分析 CrossEncoder
  python analyze_badcases.py --model_type crossencoder \
    --ckpt ../outputs/checkpoints/crossencoder_best.pt

  # 分析更多 bad case
  python analyze_badcases.py --n_cases 20

依赖：
  pip install torch transformers scikit-learn matplotlib
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset import PairDataset, CrossEncoderDataset, load_jsonl
from model import build_biencoder, build_crossencoder

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "afqmc"
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
FIG_DIR = ROOT / "outputs" / "figures"


# ── 收集预测结果 ──────────────────────────────────────────────────────────

@torch.no_grad()
def collect_biencoder_preds(model, loader, raw_rows, device, threshold):
    """返回每条样本的 {sentence1, sentence2, label, sim, pred} 列表"""
    model.eval()
    results = []
    idx = 0
    for batch in loader:
        batch_a = {
            "input_ids": batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        }
        batch_b = {
            "input_ids": batch["input_ids_b"].to(device),
            "attention_mask": batch["attention_mask_b"].to(device),
            "token_type_ids": batch["token_type_ids_b"].to(device),
        }
        emb_a, emb_b = model(batch_a, batch_b)
        sims = f.cosine_similarity(emb_a, emb_b, dim=-1).cpu().tolist()
        labels = batch["label"].tolist()

        for sim, label in zip(sims, labels):
            row = raw_rows[idx]
            results.append({
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "label": label,
                "score": sim,
                "pred": int(sim >= threshold),
            })
            idx += 1
    return results


@torch.no_grad()
def collect_crossencoder_preds(model, loader, raw_rows, device):
    model.eval()
    results = []
    idx = 0
    for batch in loader:
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["token_type_ids"].to(device),
        ).cpu()
        probs = torch.softmax(logits, dim=-1)[:, 1].tolist()
        preds = logits.argmax(dim=-1).tolist()
        labels = batch["label"].tolist()

        for prob, pred, label in zip(probs, preds, labels):
            row = raw_rows[idx]
            results.append({
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "label": label,
                "score": prob,  # P(相似)
                "pred": pred,
            })
            idx += 1
    return results


# ── Bad Case 分类 ─────────────────────────────────────────────────────────

def split_badcases(results, threshold=0.5):
    """
    分类 bad case：
      FP（假阳性）：pred=1, label=0  → 模型过度自信"相似"
      FN（假阴性）：pred=0, label=1  → 模型错过真实相似对

    同时按置信度分级：
      high confidence error : score 远离 threshold（偏差 > 0.15）
      borderline error      : score 接近 threshold（偏差 ≤ 0.15）
    """
    fp_high, fp_border = [], []
    fn_high, fn_border = [], []

    for r in results:
        if r["pred"] == r["label"]:
            continue
        gap = abs(r["score"] - threshold)
        if r["pred"] == 1 and r["label"] == 0:  # FP
            (fp_high if gap > 0.15 else fp_border).append(r)
        else:  # FN
            (fn_high if gap > 0.15 else fn_border).append(r)

    # 按置信度排序（最错的在前）
    fp_high.sort(key=lambda x: x["score"], reverse=True)
    fn_high.sort(key=lambda x: x["score"])

    return {
        "fp_high": fp_high,
        "fp_border": fp_border,
        "fn_high": fn_high,
        "fn_border": fn_border,
    }


# ── 模式分析 ──────────────────────────────────────────────────────────────

def analyze_patterns(cases, label):
    """对一批 bad case 做词汇重叠、长度差等简单统计"""
    if not cases:
        return
    len_diffs = [abs(len(r["sentence1"]) - len(r["sentence2"])) for r in cases]
    lens_s1 = [len(r["sentence1"]) for r in cases]
    lens_s2 = [len(r["sentence2"]) for r in cases]

    # 字符级 Jaccard 相似度（用 unigram 集合）
    def jaccard(a, b):
        sa, sb = set(a), set(b)
        return len(sa & sb) / max(len(sa | sb), 1)

    jaccards = [jaccard(r["sentence1"], r["sentence2"]) for r in cases]

    print(f"\n  【{label}】共 {len(cases)} 条")
    print(f"    长度差     : 均值={np.mean(len_diffs):.1f}  中位={np.median(len_diffs):.0f}")
    print(f"    s1 长度    : 均值={np.mean(lens_s1):.1f}")
    print(f"    s2 长度    : 均值={np.mean(lens_s2):.1f}")
    print(f"    字符 Jaccard: 均值={np.mean(jaccards):.3f}  "
          f"（1=完全重叠，0=无共同字符）")


def print_cases(cases, title, n=5):
    print(f"\n  {title} (展示 {min(n, len(cases))} 条)：")
    for r in cases[:n]:
        score_str = f"score={r['score']:.3f}"
        print(f"    {score_str}  | {r['sentence1']!r}")
        print(f"    {'':>12}  | {r['sentence2']!r}")
        print()


# ── Score 分布可视化 ──────────────────────────────────────────────────────

def plot_score_dist_with_errors(results, threshold, save_path, model_label):
    """
    正确 vs 错误预测的分数分布——观察错误集中在哪个 score 区间
    """
    scores = np.array([r["score"] for r in results])
    labels = np.array([r["label"] for r in results])
    correct = np.array([r["pred"] == r["label"] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：正/负样本分数分布（同 evaluate.py，基准参考）
    ax = axes[0]
    ax.hist(scores[labels == 1], bins=40, alpha=0.6, label="positive", color="#2196F3", density=True)
    ax.hist(scores[labels == 0], bins=40, alpha=0.6, label="negative", color="#F44336", density=True)
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.2f}")
    ax.set_xlabel("Score (cosine sim / P(similar))")
    ax.set_title(f"[{model_label}] Positive vs Negative Distribution")
    ax.legend(fontsize=8)

    # 右图：正确 vs 错误预测的分数分布
    ax = axes[1]
    ax.hist(scores[correct], bins=40, alpha=0.6, label="correct", color="#4CAF50", density=True)
    ax.hist(scores[~correct], bins=40, alpha=0.6, label="error", color="#F44336", density=True)
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--")
    ax.set_xlabel("Score")
    ax.set_title(f"[{model_label}] Correct vs Error Score Distribution")
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


# ── 优化方向输出 ──────────────────────────────────────────────────────────

def print_optimization_directions(badcases, model_type, fp_jaccard_mean, fn_jaccard_mean):
    """
    根据 bad case 的实际模式，输出有针对性的优化方向。
    """
    print(f"\n{'=' * 60}")
    print("优化方向建议（基于当前 bad case 分析）")
    print(f"{'=' * 60}")

    print("""
【1】数据层面
  ├─ 难负样本增强（Hard Negative Mining）
  │    当前负样本是随机采样，FP 案例中很多是"话题相关但语义不同"的句对。
  │    → 用训练好的 BiEncoder 在大规模数据中挖掘相似度高但标签为 0 的对，
  │      加入训练集，提升负例的区分难度。
  │
  ├─ 数据增强（正样本扩充）
  │    AFQMC 正样本只有 10K 条（31%），TripletLoss 三元组因此受限。
  │    → 对正样本做同义改写（换词、调序），扩充正例数量。
  │      可用 LLM API 批量生成改写句。
  │
  └─ 跨数据集迁移
       LCQMC / BQ Corpus（已下载）包含更多样的问句对。
       → 先在 LCQMC（238K 对）上预训练 Sentence-BERT，再 fine-tune 到 AFQMC，
         利用大数据集的语义泛化能力。
""")

    if fp_jaccard_mean > 0.5:
        print("""【2】模型层面（针对 FP：字符重叠高但语义不同）
  ├─ 加大 CosineEmbeddingLoss 的 margin（0.3 → 0.5）
  │    让模型对负例要求更严格，不只是让余弦相似度 < margin，
  │    而是要推得更远，减少高词汇重叠带来的假阳性。
  │
  └─ 引入 Interaction 特征辅助
       BiEncoder 看不到词序交互，而这类 FP 往往需要理解"借呗还款"vs"借呗账单"
       的细粒度区别。可在顶层加一个轻量级 Cross-Attention（保留向量化能力）。
""")
    else:
        print("""【2】模型层面（FP 字符重叠不高，主要是语义理解不足）
  ├─ 增加 BERT 层数（4 → 8 → 12 层）
  │    浅层 BERT 对语义的建模能力有限，更深层能捕捉更细粒度的语义差异。
  │
  └─ 换用金融领域预训练模型
       FinBERT / MacBERT / RoBERTa-Chinese 在金融/客服语料上有更好的初始化，
       AFQMC 中很多错误源于领域术语理解不准确。
""")

    if fn_jaccard_mean < 0.3:
        print("""【3】训练策略层面（针对 FN：词汇重叠低但语义相同的同义句）
  ├─ SimCSE 对比学习预训练
  │    同一句话 dropout 两次得到两个正例，大 batch 内其他句子为负例。
  │    这种方式能让模型学到"用不同词说同一个意思"的不变性。
  │
  └─ 调小 TripletLoss 的 margin（0.3 → 0.1）
       如果正例本身语义就不太相似（同义但换了词），过大的 margin 反而
       要求 sim(a,p) 比 sim(a,n) 高出太多，训练信号消失。
""")

    print("""【4】评估与部署层面
  ├─ 阈值校准
  │    当前阈值在 val 集上网格搜索，但 val 正负比（31:69）和线上分布可能不同。
  │    → 收集真实线上日志，按实际分布重新校准阈值（Platt scaling 等）。
  │
  ├─ 两阶段级联（最实用的工程改进）
  │    BiEncoder（召回 Top-K）→ CrossEncoder（精排 Top-1）
  │    这是 rag_annual_report Reranker 的完整版。
  │    → 可用当前两个 checkpoint 直接组合，无需重新训练。
  │
  └─ 训练更多 epoch + 全量 12 层
       本次演示用 4 层 × 1 epoch 快速验证，完整训练预计提升 5~10 个 F1 点。
       → 建议学生实验：4 层 vs 12 层，1 epoch vs 5 epoch 的 2×2 消融。
""")


# ── 主流程 ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Bad Case 分析与优化方向")
    parser.add_argument("--model_type", default="biencoder",
                        choices=["biencoder", "crossencoder"])
    parser.add_argument("--ckpt", default=None, type=str,
                        help="checkpoint 路径（默认自动选 biencoder_cosine_best.pt）")
    parser.add_argument("--split", default="validation",
                        choices=["validation", "test"])
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--n_cases", default=5, type=int,
                        help="每类 bad case 展示条数")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 确定 checkpoint 路径 ──────────────────────────────────────────────
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    elif args.model_type == "biencoder":
        ckpt_path = CKPT_DIR / "biencoder_cosine_best.pt"
    else:
        ckpt_path = CKPT_DIR / "crossencoder_best.pt"

    if not ckpt_path.exists():
        print(f"❌ checkpoint 不存在: {ckpt_path}")
        print("请先运行训练脚本（train_biencoder.py / train_crossencoder.py）")
        return

    print(f"加载 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    # ── 加载原始数据和 DataLoader ─────────────────────────────────────────
    data_path = DATA_DIR / f"{args.split}.jsonl"
    raw_rows = load_jsonl(data_path)
    print(f"数据集: {data_path.name}  共 {len(raw_rows):,} 条")

    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))

    # ── 收集预测结果 ──────────────────────────────────────────────────────
    if args.model_type == "biencoder":
        model = build_biencoder(
            bert_path=str(BERT_PATH),
            pool=saved_args.get("pool", "mean"),
            num_hidden_layers=saved_args.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        threshold = ckpt.get("threshold", 0.5)

        ds = PairDataset(data_path, tokenizer, saved_args.get("max_length", 64))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        results = collect_biencoder_preds(model, loader, raw_rows, device, threshold)
        model_label = f"BiEncoder(loss={saved_args.get('loss', 'cosine')},threshold={threshold:.2f})"

    else:
        model = build_crossencoder(
            bert_path=str(BERT_PATH),
            num_hidden_layers=saved_args.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        threshold = 0.5  # P(相似)=0.5 作为分界

        ds = CrossEncoderDataset(data_path, tokenizer, max_length=128)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        results = collect_crossencoder_preds(model, loader, raw_rows, device)
        model_label = f"CrossEncoder(threshold=P≥0.5)"

    # ── 整体指标 ──────────────────────────────────────────────────────────
    labels = [r["label"] for r in results]
    preds = [r["pred"] for r in results]
    n_correct = sum(1 for l, p in zip(labels, preds) if l == p)
    accuracy = n_correct / len(results)
    print(f"\n整体准确率: {accuracy:.4f}  错误数: {len(results) - n_correct}")

    # ── 分类 bad case ─────────────────────────────────────────────────────
    badcases = split_badcases(results, threshold)
    fp_all = badcases["fp_high"] + badcases["fp_border"]
    fn_all = badcases["fn_high"] + badcases["fn_border"]

    print(f"\n{'=' * 60}")
    print(f"Bad Case 汇总  (共 {len(fp_all) + len(fn_all)} 个错误)")
    print(f"{'─' * 60}")
    print(f"  FP 假阳性（预测相似，实际不同）: {len(fp_all):>4} 条")
    print(f"    其中高置信度错误  (Δscore>0.15): {len(badcases['fp_high']):>3} 条")
    print(f"    其中临界错误     (Δscore≤0.15): {len(badcases['fp_border']):>3} 条")
    print(f"  FN 假阴性（预测不同，实际相似）: {len(fn_all):>4} 条")
    print(f"    其中高置信度错误  (Δscore>0.15): {len(badcases['fn_high']):>3} 条")
    print(f"    其中临界错误     (Δscore≤0.15): {len(badcases['fn_border']):>3} 条")

    # ── 模式分析 ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Bad Case 语言特征分析：")
    analyze_patterns(fp_all, "FP（假阳性）")
    analyze_patterns(fn_all, "FN（假阴性）")

    fp_jaccard_mean = (np.mean([
        len(set(r["sentence1"]) & set(r["sentence2"])) /
        max(len(set(r["sentence1"]) | set(r["sentence2"])), 1)
        for r in fp_all
    ]) if fp_all else 0)
    fn_jaccard_mean = (np.mean([
        len(set(r["sentence1"]) & set(r["sentence2"])) /
        max(len(set(r["sentence1"]) | set(r["sentence2"])), 1)
        for r in fn_all
    ]) if fn_all else 0)

    # ── 展示典型案例 ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print_cases(badcases["fp_high"], f"FP 高置信度错误（score最高的{args.n_cases}条）",
                n=args.n_cases)
    print_cases(badcases["fp_border"], f"FP 临界错误（{args.n_cases}条）",
                n=args.n_cases)
    print_cases(badcases["fn_high"], f"FN 高置信度错误（score最低的{args.n_cases}条）",
                n=args.n_cases)
    print_cases(badcases["fn_border"], f"FN 临界错误（{args.n_cases}条）",
                n=args.n_cases)

    # ── 可视化 ────────────────────────────────────────────────────────────
    fig_path = FIG_DIR / f"{args.model_type}_badcase_dist.png"
    plot_score_dist_with_errors(results, threshold, fig_path, model_label)

    # ── 优化方向 ──────────────────────────────────────────────────────────
    print_optimization_directions(badcases, args.model_type,
                                  fp_jaccard_mean, fn_jaccard_mean)


if __name__ == "__main__":
    main()
