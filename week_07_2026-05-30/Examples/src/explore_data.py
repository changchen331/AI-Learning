"""
cluener2020 数据集探索与可视化

教学重点：
  1. span 标注格式的统计方法（与 BIO 格式的信息等价）
  2. 各实体类型的分布差异（为什么类别不均衡是NER的难点）
  3. 文本长度分布（影响 BERT max_length 的选择）
  4. 实体长度分布（短实体 vs 长实体的识别难度差异）

使用方式：
  python explore_data.py

输出：
  outputs/figures/entity_distribution.png   各类实体频次分布
  outputs/figures/text_length_distribution.png  文本长度分布
  outputs/figures/entity_length_distribution.png 实体长度分布
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "cluener"
FIG_DIR = ROOT / "outputs" / "figures"


def load_split(split: str) -> list:
    path = DATA_DIR / f"{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_stats(records: list) -> dict:
    entity_type_counts = Counter()
    entity_lengths = []
    text_lengths = []
    entity_per_sentence = []
    entities_by_type = {}

    for row in records:
        text = row["text"]
        text_lengths.append(len(text))
        label = row.get("label") or {}

        total_entities = 0
        for etype, spans in label.items():
            for surface, positions in spans.items():
                for start, end in positions:
                    entity_type_counts[etype] += 1
                    entity_lengths.append(end - start + 1)
                    total_entities += 1
                    if etype not in entities_by_type:
                        entities_by_type[etype] = []
                    entities_by_type[etype].append(surface)

        entity_per_sentence.append(total_entities)

    return {
        "entity_type_counts": entity_type_counts,
        "entity_lengths": entity_lengths,
        "text_lengths": text_lengths,
        "entity_per_sentence": entity_per_sentence,
        "entities_by_type": entities_by_type,
    }


def print_summary(stats_train: dict, stats_val: dict):
    print("=" * 70)
    print("cluener2020 数据集统计摘要")
    print("=" * 70)

    print("\n【训练集】")
    print(f"  样本数：{len(stats_train['text_lengths'])} 条")
    print(f"  文本平均长度：{sum(stats_train['text_lengths']) / len(stats_train['text_lengths']):.1f} 字")
    print(f"  文本最大长度：{max(stats_train['text_lengths'])} 字")
    print(f"  文本长度中位数：{sorted(stats_train['text_lengths'])[len(stats_train['text_lengths']) // 2]} 字")
    print(f"  平均实体数/句：{sum(stats_train['entity_per_sentence']) / len(stats_train['entity_per_sentence']):.2f}")
    print(f"  实体总数：{sum(stats_train['entity_type_counts'].values())}")
    print(f"  平均实体长度：{sum(stats_train['entity_lengths']) / len(stats_train['entity_lengths']):.1f} 字")

    print("\n【各类实体频次（训练集）】")
    et_label = {
        "address": "地址", "book": "书名", "company": "公司",
        "game": "游戏", "government": "政府机构", "movie": "影视作品",
        "name": "人名", "organization": "组织机构", "position": "职位",
        "scene": "景点/场所",
    }
    for etype, cnt in sorted(stats_train["entity_type_counts"].items(), key=lambda x: -x[1]):
        cn = et_label.get(etype, etype)
        print(f"  {etype:15s} ({cn:8s}) : {cnt:5d} 条")

    print("\n【各类实体示例（训练集，取前5个）】")
    for etype in sorted(stats_train["entities_by_type"]):
        cn = et_label.get(etype, etype)
        examples = list(dict.fromkeys(stats_train["entities_by_type"][etype]))[:5]
        print(f"  {etype:15s} ({cn}) : {' | '.join(examples)}")

    print()


def plot_entity_distribution(stats_train: dict):
    et_label = {
        "address": "地址", "book": "书名", "company": "公司",
        "game": "游戏", "government": "政府", "movie": "影视",
        "name": "人名", "organization": "组织", "position": "职位",
        "scene": "景点",
    }
    counts = stats_train["entity_type_counts"]
    labels = [f"{k}\n({et_label.get(k, k)})" for k in sorted(counts)]
    values = [counts[k] for k in sorted(counts)]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, values, color="#4C72B0", alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20, str(v),
                ha="center", va="bottom", fontsize=9)
    ax.set_title("cluener2020 各类实体频次分布（训练集）", fontsize=14)
    ax.set_ylabel("实体数量")
    ax.set_xlabel("实体类型")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "entity_distribution.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / 'entity_distribution.png'}")
    plt.close()


def plot_text_length_distribution(stats_train: dict):
    lengths = stats_train["text_lengths"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths, bins=40, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(x=64, color="red", linestyle="--", linewidth=1.5, label="max_length=64")
    ax.axvline(x=128, color="orange", linestyle="--", linewidth=1.5, label="max_length=128")
    p95 = sorted(lengths)[int(len(lengths) * 0.95)]
    ax.axvline(x=p95, color="green", linestyle="--", linewidth=1.5, label=f"P95={p95}")
    ax.set_title("文本长度分布（训练集）", fontsize=14)
    ax.set_xlabel("文本字符数")
    ax.set_ylabel("样本数")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "text_length_distribution.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / 'text_length_distribution.png'}")
    plt.close()
    print(f"  P95 文本长度={p95}，建议 max_length=128")


def plot_entity_length_distribution(stats_train: dict):
    from collections import Counter
    lengths = Counter(stats_train["entity_lengths"])
    xs = sorted(lengths.keys())
    ys = [lengths[x] for x in xs]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([str(x) for x in xs[:20]], ys[:20], color="#55A868", alpha=0.85, edgecolor="white")
    ax.set_title("实体长度分布（训练集，前20）", fontsize=14)
    ax.set_xlabel("实体字符数")
    ax.set_ylabel("出现次数")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "entity_length_distribution.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / 'entity_length_distribution.png'}")
    plt.close()

    avg_len = sum(stats_train["entity_lengths"]) / len(stats_train["entity_lengths"])
    print(f"  实体平均长度={avg_len:.1f}字，CRF 对短实体边界识别优势更明显")


def main():
    parse_args()

    train_records = load_split("train")
    val_records = load_split("validation")

    stats_train = collect_stats(train_records)
    stats_val = collect_stats(val_records)

    print_summary(stats_train, stats_val)

    print("正在生成可视化图表...")
    plot_entity_distribution(stats_train)
    plot_text_length_distribution(stats_train)
    plot_entity_length_distribution(stats_train)

    print("\n探索完成！图表已保存到 outputs/figures/")
    print("下一步：python train.py               # 训练 BERT+Linear")
    print("         python train.py --use_crf    # 训练 BERT+CRF")


def parse_args():
    parser = argparse.ArgumentParser(description="探索 cluener2020 数据集")
    return parser.parse_args()


if __name__ == "__main__":
    main()
