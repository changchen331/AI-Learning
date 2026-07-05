"""
AFQMC 数据集探索与可视化

教学重点：
  1. 文本匹配数据的结构——sentence pair + binary label，与分类任务的本质区别
  2. 类别不均衡（~31% 正例）——实际业务中"不同义"的问题对比"同义"更多见
  3. 句子长度分布——BERT max_length 截断阈值的选择依据
  4. 正/负样本的长度差异——是否存在"长句倾向于不相似"的捷径（shortcut）
  5. Token 数 vs 字符数——BERT 中文字节对编码的粒度

使用方式：
  python explore_data.py
  python explore_data.py --data_dir ../data/afqmc --output_dir ../outputs/figures

依赖：
  pip install matplotlib transformers
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from transformers import BertTokenizer

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "afqmc"
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs" / "figures"

# 中文字体（matplotlib 默认不支持中文）
_CN_FONT = None


def _get_font():
    global _CN_FONT
    if _CN_FONT is None:
        try:
            font_path = next(
                p for p in fm.findSystemFonts()
                if any(k in p.lower() for k in ("simhei", "msyh", "simsun", "notosans"))
            )
            _CN_FONT = fm.FontProperties(fname=font_path)
        except StopIteration:
            _CN_FONT = fm.FontProperties()
    return _CN_FONT


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── 图 1：标签分布 ────────────────────────────────────────────────────────

def plot_label_distribution(splits_data, output_dir):
    fig, axes = plt.subplots(1, len(splits_data), figsize=(10, 4))
    if len(splits_data) == 1:
        axes = [axes]

    fp = _get_font()
    for ax, (split_name, rows) in zip(axes, splits_data.items()):
        labels = [r["label"] for r in rows]
        cnt = Counter(labels)
        counts = [cnt.get(0, 0), cnt.get(1, 0)]
        bars = ax.bar(["不相似 (0)", "相似 (1)"], counts,
                      color=["#F44336", "#2196F3"], width=0.5)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    f"{c}\n({c / len(rows) * 100:.1f}%)", ha="center", va="bottom",
                    fontproperties=fp, fontsize=9)
        ax.set_title(f"{split_name}（{len(rows):,} 条）", fontproperties=fp)
        ax.set_ylabel("数量", fontproperties=fp)
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle("AFQMC 标签分布（正例约 31%，负例约 69%）", fontproperties=fp,
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save_path = output_dir / "label_distribution.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


# ── 图 2：句子字符长度分布 ────────────────────────────────────────────────

def plot_char_length(rows, output_dir):
    pos_rows = [r for r in rows if r["label"] == 1]
    neg_rows = [r for r in rows if r["label"] == 0]

    def lens(rs):
        return [len(r["sentence1"]) for r in rs] + [len(r["sentence2"]) for r in rs]

    pos_lens = lens(pos_rows)
    neg_lens = lens(neg_rows)

    fp = _get_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pos_lens, bins=40, alpha=0.6, label="正样本（相似）",
            color="#2196F3", density=True)
    ax.hist(neg_lens, bins=40, alpha=0.6, label="负样本（不相似）",
            color="#F44336", density=True)
    ax.axvline(32, color="black", linestyle="--", linewidth=1,
               label="max_length=32")
    ax.axvline(64, color="gray", linestyle="--", linewidth=1,
               label="max_length=64")
    ax.set_xlabel("句子字符长度", fontproperties=fp)
    ax.set_ylabel("密度", fontproperties=fp)
    ax.set_title("正/负样本句子长度分布（train）", fontproperties=fp)
    ax.legend(prop=fp)
    fig.tight_layout()

    save_path = output_dir / "char_length_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")

    all_lens = [len(r["sentence1"]) for r in rows] + [len(r["sentence2"]) for r in rows]
    print(f"  字符长度统计（train 全部句子）：")
    print(f"    均值={np.mean(all_lens):.1f}  中位数={np.median(all_lens):.0f}  "
          f"P95={np.percentile(all_lens, 95):.0f}  最长={max(all_lens)}")
    for threshold in [32, 48, 64, 96]:
        cover = sum(1 for l in all_lens if l <= threshold) / len(all_lens) * 100
        print(f"    max_length={threshold:3d} 覆盖率: {cover:.1f}%")


# ── 图 3：Token 数分布（BERT Tokenizer） ─────────────────────────────────

def plot_token_length(rows, tokenizer, output_dir):
    print("  计算 Token 长度（需要 tokenize，稍慢...）")
    token_lens = []
    for r in rows[:5000]:  # 取前 5000 条避免太慢
        t1 = len(tokenizer.tokenize(r["sentence1"]))
        t2 = len(tokenizer.tokenize(r["sentence2"]))
        token_lens.extend([t1, t2])

    fp = _get_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(token_lens, bins=40, color="#4CAF50", alpha=0.8, density=True)
    ax.axvline(np.mean(token_lens), color="red", linestyle="-",
               label=f"均值={np.mean(token_lens):.1f}")
    ax.axvline(np.percentile(token_lens, 95), color="orange", linestyle="--",
               label=f"P95={np.percentile(token_lens, 95):.0f}")
    ax.set_xlabel("单句 Token 数（不含 [CLS]/[SEP]）", fontproperties=fp)
    ax.set_ylabel("密度", fontproperties=fp)
    ax.set_title("单句 Token 数分布（train 前 5000 条）", fontproperties=fp)
    ax.legend(prop=fp)
    fig.tight_layout()

    save_path = output_dir / "token_length_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")
    print(f"  Token 长度：均值={np.mean(token_lens):.1f}  "
          f"P95={np.percentile(token_lens, 95):.0f}  最长={max(token_lens)}")


# ── 图 4：正/负样本长度差（捷径检测） ────────────────────────────────────

def plot_length_diff(rows, output_dir):
    """
    检测长度差是否可作为"判别捷径"：
    若正样本句子长度差 << 负样本，则模型可能学到"长度接近 → 相似"这个捷径。
    教学价值：启发学生思考数据集偏差对模型泛化的影响。
    """
    pos_diffs = [abs(len(r["sentence1"]) - len(r["sentence2"]))
                 for r in rows if r["label"] == 1]
    neg_diffs = [abs(len(r["sentence1"]) - len(r["sentence2"]))
                 for r in rows if r["label"] == 0]

    fp = _get_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pos_diffs, bins=30, alpha=0.6, label=f"正样本 均值={np.mean(pos_diffs):.1f}",
            color="#2196F3", density=True)
    ax.hist(neg_diffs, bins=30, alpha=0.6, label=f"负样本 均值={np.mean(neg_diffs):.1f}",
            color="#F44336", density=True)
    ax.set_xlabel("|len(s1) - len(s2)| 字符数", fontproperties=fp)
    ax.set_ylabel("密度", fontproperties=fp)
    ax.set_title("正/负样本句子长度差分布（length bias 检测）", fontproperties=fp)
    ax.legend(prop=fp)
    fig.tight_layout()

    save_path = output_dir / "length_diff_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")
    print(f"  长度差：正样本均值={np.mean(pos_diffs):.1f}  负样本均值={np.mean(neg_diffs):.1f}")
    if np.mean(pos_diffs) < np.mean(neg_diffs) * 0.7:
        print("  ⚠️  正样本长度差明显更小，存在 length bias 风险")
    else:
        print("  ✓ 正/负样本长度差接近，无明显 length bias")


# ── 控制台统计输出 ────────────────────────────────────────────────────────

def print_stats(name, rows):
    labels = [r["label"] for r in rows]
    cnt = Counter(labels)
    s1_lens = [len(r["sentence1"]) for r in rows]
    s2_lens = [len(r["sentence2"]) for r in rows]
    all_lens = s1_lens + s2_lens

    print(f"\n{'=' * 50}")
    print(f"【{name}】共 {len(rows):,} 条")
    print(f"{'=' * 50}")

    n_pos = cnt.get(1, 0)
    n_neg = cnt.get(0, 0)
    n_unlabeled = sum(v for k, v in cnt.items() if k not in (0, 1))
    if n_unlabeled:
        # AFQMC 竞赛测试集标签为 -1（未公开），不计入正/负统计
        print(f"  标签未公开（CLUE 竞赛格式）: {n_unlabeled:>6,} 条  —— 仅供参考，不用于评估")
    else:
        print(f"  正样本（相似）  : {n_pos:>6,} ({n_pos / len(rows) * 100:.1f}%)")
        print(f"  负样本（不相似）: {n_neg:>6,} ({n_neg / len(rows) * 100:.1f}%)")
        print(f"  不均衡比 (neg/pos): {n_neg / max(n_pos, 1):.1f}x")
    print(f"  句子字符长度 — 均值={np.mean(all_lens):.1f}  中位数={np.median(all_lens):.0f}  "
          f"P95={np.percentile(all_lens, 95):.0f}  最长={max(all_lens)}")
    print(f"  示例正样本：")
    for r in [r for r in rows if r["label"] == 1][:2]:
        print(f"    ✓  {r['sentence1']!r}  ||  {r['sentence2']!r}")
    print(f"  示例负样本：")
    for r in [r for r in rows if r["label"] == 0][:2]:
        print(f"    ✗  {r['sentence1']!r}  ||  {r['sentence2']!r}")


# ── 主流程 ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="AFQMC 数据集探索")
    parser.add_argument("--data_dir", default=str(DATA_DIR), type=Path)
    parser.add_argument("--bert_path", default=str(BERT_PATH), type=str)
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), type=Path)
    parser.add_argument("--skip_token", action="store_true", help="跳过 Token 长度分析（较慢）")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for split in ["train", "validation", "test"]:
        path = args.data_dir / f"{split}.jsonl"
        if path.exists():
            splits[split] = load_jsonl(path)

    for name, rows in splits.items():
        print_stats(name, rows)

    train_rows = splits.get("train", [])
    if not train_rows:
        print("train.jsonl 不存在，请先运行 download_data.py")
        return

    print(f"\n{'=' * 50}")
    print("生成可视化图表...")

    plot_label_distribution(splits, args.output_dir)
    plot_char_length(train_rows, args.output_dir)
    plot_length_diff(train_rows, args.output_dir)

    if not args.skip_token:
        tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        plot_token_length(train_rows, tokenizer, args.output_dir)

    print(f"\n所有图表已保存至 → {args.output_dir}")


if __name__ == "__main__":
    main()
