"""
人民日报 NER：LLM API zero-shot vs few-shot 对比

与 cluener 版本的差异：
  1. 实体类型从 10 类（cluener）→ 3 类（PER/ORG/LOC，人民日报）
  2. 数据格式：tokens + ner_tags（已是 BIO），需要把 BIO 还原成 (surface, type, start, end)
     而 cluener 是 span 字典，可直接读出 (surface, type, start, end)

使用方式：
  python llm_ner.py
  python llm_ner.py --n_samples 100 --model qwen-plus

依赖：
  pip install openai
  set DASHSCOPE_API_KEY=sk-xxx
"""

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

EXP_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = EXP_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data" / "peoples_daily"
LOG_DIR = EXP_ROOT / "outputs" / "logs"

ENTITY_TYPE_ZH = {
    "PER": "人名",
    "ORG": "组织机构",
    "LOC": "地点",
}

ENTITY_TYPES = list(ENTITY_TYPE_ZH.keys())


def build_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def bio_to_spans(tokens: list[str], tags: list[str]) -> set[tuple[str, str, int, int]]:
    """从 BIO 标签序列还原 (surface, type, start, end) 4 元组集合。

    遍历 tags：
      - 遇到 B-X：开始一个新实体
      - 遇到 I-X 且与当前类型一致：扩展实体
      - 否则：结束当前实体并开新实体或跳过
    返回的 (start, end) 是字符索引（含两端）。
    """
    spans = set()
    i = 0
    n = len(tags)
    while i < n:
        tag = tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            start = i
            j = i + 1
            while j < n and tags[j] == f"I-{etype}":
                j += 1
            end = j - 1
            surface = "".join(tokens[start:end + 1])
            spans.add((surface, etype, start, end))
            i = j
        else:
            i += 1
    return spans


def gold_spans_from_record(record: dict) -> set[tuple[str, str, int, int]]:
    return bio_to_spans(record["tokens"], record["ner_tags"])


def pred_spans_from_response(text: str, response_text: str) -> set[tuple[str, str, int, int]]:
    """从 LLM 输出中解析实体（与 cluener 版完全一致的解析逻辑）。"""
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        return set()

    try:
        obj = json.loads(json_match.group())
    except json.JSONDecodeError:
        return set()

    entities = obj.get("entities", [])
    if not isinstance(entities, list):
        return set()

    spans = set()
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        surface = str(ent.get("text", "")).strip()
        etype = str(ent.get("type", "")).strip().upper()
        if not surface or etype not in ENTITY_TYPES:
            continue
        idx = text.find(surface)
        if idx == -1:
            continue
        spans.add((surface, etype, idx, idx + len(surface) - 1))

    return spans


def compute_span_f1(all_golds: list[set], all_preds: list[set]) -> dict:
    tp = sum(len(g & p) for g, p in zip(all_golds, all_preds))
    pred_total = sum(len(p) for p in all_preds)
    gold_total = sum(len(g) for g in all_golds)
    p = tp / pred_total if pred_total else 0.0
    r = tp / gold_total if gold_total else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "pred_total": pred_total, "gold_total": gold_total}


SYSTEM_PROMPT = """你是一个命名实体识别（NER）专家，专门处理中文文本。
请从用户输入的文本中识别以下 3 类实体，并以 JSON 格式输出结果：
- PER：人名
- ORG：组织机构（公司、机构、团体等）
- LOC：地点（国家、省份、城市、地区等）

输出格式（严格遵守，不要包含其他文字）：
{"entities": [{"text": "实体文本", "type": "实体类型大写英文"}, ...]}

如果没有实体，输出：{"entities": []}"""

FEW_SHOT_EXAMPLES = [
    {
        "text": "海钓比赛地点在厦门与金门之间的海域。",
        "output": '{"entities": [{"text": "厦门", "type": "LOC"}, {"text": "金门", "type": "LOC"}]}'
    },
    {
        "text": "美国国务卿蓬佩奥访问了北京，并与外交部长王毅举行会谈。",
        "output": '{"entities": [{"text": "美国", "type": "LOC"}, {"text": "蓬佩奥", "type": "PER"}, {"text": "北京", "type": "LOC"}, {"text": "外交部", "type": "ORG"}, {"text": "王毅", "type": "PER"}]}'
    },
    {
        "text": "中国科学院在上海召开了关于人工智能的研讨会。",
        "output": '{"entities": [{"text": "中国科学院", "type": "ORG"}, {"text": "上海", "type": "LOC"}]}'
    },
]


def zero_shot_prompt(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def few_shot_prompt(text: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["text"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": text})
    return messages


def call_api(client: OpenAI, messages: list[dict], model: str) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  API 调用失败：{e}")
                return ""
    return ""


def sample_records(n: int, seed: int = 42) -> list[dict]:
    """从验证集中采样，保证 PER/ORG/LOC 三类都覆盖。"""
    with open(DATA_DIR / "validation.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    random.seed(seed)
    by_type = defaultdict(list)
    for r in records:
        types = set()
        for t in r["ner_tags"]:
            if t.startswith("B-"):
                types.add(t[2:])
        for et in types:
            by_type[et].append(r)

    selected = set()
    selected_list = []
    per_type = max(1, n // len(ENTITY_TYPES))
    for etype in ENTITY_TYPES:
        candidates = [r for r in by_type[etype] if id(r) not in selected]
        chosen = random.sample(candidates, min(per_type, len(candidates)))
        for r in chosen:
            if len(selected_list) < n and id(r) not in selected:
                selected.add(id(r))
                selected_list.append(r)

    remaining = [r for r in records if id(r) not in selected]
    random.shuffle(remaining)
    for r in remaining:
        if len(selected_list) >= n:
            break
        selected_list.append(r)

    return selected_list[:n]


def main():
    args = parse_args()

    client = build_client()
    records = sample_records(args.n_samples)
    print(f"采样 {len(records)} 条验证集样本")

    zero_shot_golds = []
    zero_shot_preds = []
    few_shot_golds = []
    few_shot_preds = []
    detail_records = []

    for i, record in enumerate(records, 1):
        text = "".join(record["tokens"])
        gold = gold_spans_from_record(record)

        zs_resp = call_api(client, zero_shot_prompt(text), args.model)
        zs_pred = pred_spans_from_response(text, zs_resp)

        fs_resp = call_api(client, few_shot_prompt(text), args.model)
        fs_pred = pred_spans_from_response(text, fs_resp)

        zero_shot_golds.append(gold)
        zero_shot_preds.append(zs_pred)
        few_shot_golds.append(gold)
        few_shot_preds.append(fs_pred)

        detail_records.append({
            "text": text,
            "gold": [{"text": s, "type": t} for s, t, _, _ in gold],
            "zero_shot": [{"text": s, "type": t} for s, t, _, _ in zs_pred],
            "few_shot": [{"text": s, "type": t} for s, t, _, _ in fs_pred],
        })

        if i % 10 == 0 or i == len(records):
            print(f"  已处理 {i}/{len(records)} 条")

    zs_metrics = compute_span_f1(zero_shot_golds, zero_shot_preds)
    fs_metrics = compute_span_f1(few_shot_golds, few_shot_preds)

    print("\n" + "=" * 60)
    print(f"LLM NER 对比结果（模型：{args.model}，样本：{len(records)} 条）")
    print("=" * 60)
    print(f"{'方案':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)
    print(f"{'Zero-shot':<20} {zs_metrics['precision']:>10.4f} {zs_metrics['recall']:>10.4f} {zs_metrics['f1']:>10.4f}")
    print(
        f"{'Few-shot (3例)':<20} {fs_metrics['precision']:>10.4f} {fs_metrics['recall']:>10.4f} {fs_metrics['f1']:>10.4f}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "model": args.model,
        "n_samples": len(records),
        "zero_shot": zs_metrics,
        "few_shot": fs_metrics,
        "detail": detail_records,
    }

    def _to_python(v):
        return v.item() if hasattr(v, "item") else v

    result["zero_shot"] = {k: _to_python(v) for k, v in result["zero_shot"].items()}
    result["few_shot"] = {k: _to_python(v) for k, v in result["few_shot"].items()}

    out_path = LOG_DIR / "eval_llm.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nLLM 评估结果已保存 → {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="LLM zero-shot/few-shot NER（人民日报数据集）")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--model", type=str, default="qwen-plus")
    return parser.parse_args()


if __name__ == "__main__":
    main()
