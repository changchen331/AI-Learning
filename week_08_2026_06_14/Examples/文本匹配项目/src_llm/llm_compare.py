"""
LLM API 文本匹配对比（DashScope）

教学重点：
  1. Zero-shot prompt 做文本匹配 vs Fine-tuned BERT 的对比
  2. LLM 的优势：无需训练，直接泛化；劣势：慢、贵、无法批量检索
  3. 为什么不跑全部验证集：LLM 调用有成本，100 条足以说明效果差异
  4. 如何设计 prompt：让模型只输出"是"或"否"，降低解析难度

使用方式：
  export DASHSCOPE_API_KEY="sk-xxx"
  python llm_compare.py
  python llm_compare.py --num_samples 50 --model qwen-plus

依赖：
  pip install openai
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "afqmc"

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

PROMPT_TEMPLATE = """请判断以下两个问题是否表达相同的意思。只回答"是"或"否"，不要有任何其他内容。

问题1：{s1}
问题2：{s2}

回答："""


# ── 单次 LLM 调用 ─────────────────────────────────────────────────────────

def call_llm(client, s1, s2, model):
    prompt = PROMPT_TEMPLATE.format(s1=s1, s2=s2)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        answer = resp.choices[0].message.content.strip()
        # 解析：含"是"→1，含"否"→0，其他→-1（解析失败）
        if "是" in answer:
            return 1, answer
        elif "否" in answer:
            return 0, answer
        else:
            return -1, answer  # 解析失败
    except Exception as e:
        return -1, str(e)


# ── 批量评估 ─────────────────────────────────────────────────────────────

def evaluate_llm(samples, client, model, sleep_sec=0.2):
    results = []
    parse_fail = 0

    for i, r in enumerate(samples):
        pred, raw = call_llm(client, r["sentence1"], r["sentence2"], model)
        if pred == -1:
            parse_fail += 1
        results.append({
            "sentence1": r["sentence1"],
            "sentence2": r["sentence2"],
            "label": r["label"],
            "pred": pred,
            "raw": raw,
        })
        if (i + 1) % 10 == 0:
            done = [x for x in results if x["pred"] != -1]
            if done:
                acc_so_far = sum(1 for x in done if x["pred"] == x["label"]) / len(done)
                print(f"  [{i + 1}/{len(samples)}] 当前准确率（有效预测）: {acc_so_far:.3f}  "
                      f"解析失败: {parse_fail}")
        time.sleep(sleep_sec)

    return results, parse_fail


# ── 统计指标 ──────────────────────────────────────────────────────────────

def compute_metrics(results):
    valid = [r for r in results if r["pred"] != -1]
    if not valid:
        return {"accuracy": 0.0, "f1_pos": 0.0, "n_valid": 0, "n_fail": len(results)}

    labels = [r["label"] for r in valid]
    preds = [r["pred"] for r in valid]

    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    acc = sum(1 for l, p in zip(labels, preds) if l == p) / len(valid)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    return {
        "accuracy": acc,
        "precision_pos": prec,
        "recall_pos": rec,
        "f1_pos": f1,
        "n_valid": len(valid),
        "n_fail": len(results) - len(valid),
    }


# ── 主流程 ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="LLM zero-shot 文本匹配评估")
    parser.add_argument("--data_dir", default=str(DATA_DIR), type=str)
    parser.add_argument("--split", default="validation",
                        choices=["validation", "test"])
    parser.add_argument("--num_samples", default=100, type=int,
                        help="评估样本数（默认 100，全集太贵；AFQMC val 共 4316 条）")
    parser.add_argument("--model", default="qwen-plus", type=str)
    parser.add_argument("--sleep_sec", default=0.2, type=float,
                        help="每次调用后等待时间（秒），避免触发限流")
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 未设置 DASHSCOPE_API_KEY 环境变量")
        print("   请运行：export DASHSCOPE_API_KEY='sk-xxx'")
        return

    # 读取数据
    data_path = Path(args.data_dir) / f"{args.split}.jsonl"
    rows = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # 随机采样（保持正负比例）
    pos_rows = [r for r in rows if r["label"] == 1]
    neg_rows = [r for r in rows if r["label"] == 0]
    n_pos = min(args.num_samples // 3, len(pos_rows))  # 约 1/3 正样本
    n_neg = args.num_samples - n_pos
    samples = random.sample(pos_rows, n_pos) + random.sample(neg_rows, min(n_neg, len(neg_rows)))
    random.shuffle(samples)

    print(f"数据集: {data_path.name}  样本数: {len(samples)}")
    print(f"  正样本: {n_pos}  负样本: {len(samples) - n_pos}")
    print(f"模型: {args.model}")
    print(f"\nPrompt 示例：")
    ex = samples[0]
    print(PROMPT_TEMPLATE.format(s1=ex["sentence1"], s2=ex["sentence2"]))
    print("─" * 50)

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)

    print(f"\n开始评估（共 {len(samples)} 条，预计 {len(samples) * (0.5 + args.sleep_sec):.0f}s）...")
    results, parse_fail = evaluate_llm(samples, client, args.model, args.sleep_sec)

    metrics = compute_metrics(results)

    print(f"\n{'=' * 55}")
    print(f"LLM 评估结果（{args.model}，{len(samples)} 条样本）")
    print(f"  准确率 (Accuracy)  : {metrics['accuracy']:.4f}")
    print(f"  正例精确率         : {metrics['precision_pos']:.4f}")
    print(f"  正例召回率         : {metrics['recall_pos']:.4f}")
    print(f"  正例 F1            : {metrics['f1_pos']:.4f}")
    print(f"  有效预测数         : {metrics['n_valid']}")
    print(f"  解析失败数         : {metrics['n_fail']}")

    print(f"\n{'─' * 55}")
    print("对比参考（来自 BiEncoder/CrossEncoder 训练结果，val 集全量）：")
    print("  指标            | BiEncoder | CrossEncoder | LLM zero-shot")
    print("  Accuracy        |  (见训练日志)  |  (见训练日志)  | "
          f"{metrics['accuracy']:.4f} ({len(samples)} 样本)")
    print("  推理速度        |   毫秒级       |   秒级         |  秒级+网络延迟")
    print("  可检索（向量）  |    ✓           |    ✗           |   ✗")
    print("  需要训练        |    ✓           |    ✓           |   ✗")

    # 保存失败案例
    fail_cases = [r for r in results if r["pred"] != r["label"]][:10]
    if fail_cases:
        print(f"\n前 {len(fail_cases)} 条预测错误样本：")
        for r in fail_cases:
            label_str = "相似" if r["label"] == 1 else "不相似"
            pred_str = "相似" if r["pred"] == 1 else ("不相似" if r["pred"] == 0 else "解析失败")
            print(f"  [真:{label_str} | 预:{pred_str}]")
            print(f"    {r['sentence1']!r}  ||  {r['sentence2']!r}")

    # 保存结果到文件，供 evaluate_sft.py 读取比较
    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "llm_compare_results.json"
    save_data = {
        "model": args.model,
        "n_samples": len(samples),
        "metrics": {k: float(v) if isinstance(v, float) else v
                    for k, v in metrics.items()},
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
