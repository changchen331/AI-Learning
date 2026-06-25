"""
加载 SFT checkpoint（LoRA / 全量微调），在验证集上评估文本匹配 Accuracy / F1，
与 BiEncoder / CrossEncoder / LLM API 多方对比

教学重点：
  1. 生成式文本匹配的评估：generate → 解析"相似"/"不相似" → 与 gold label 比较
  2. 与 BERT 方案（判别式）使用相同指标（Accuracy + F1），直接横向可比
  3. LoRA 自动识别：目录含 adapter_config.json → LoRA，否则 → 全量

使用方式：
  python evaluate_sft.py                              # 评估 LoRA（默认）
  python evaluate_sft.py --ckpt_dir ../outputs/sft_full_ckpt  # 评估全量微调
  python evaluate_sft.py --demo                       # 5 条示例快速演示

依赖：
  pip install torch transformers peft scikit-learn
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from peft import PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "afqmc"
MODEL_PATH = ROOT.parent.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"
ADAPTER_DIR = ROOT / "outputs" / "sft_adapter"
LOG_DIR = ROOT / "outputs" / "logs"

SYSTEM_PROMPT = (
    "你是一个语义匹配助手。判断两句话语义是否相同，"
    "只输出【相似】或【不相似】，不要输出其他内容。"
)


# ══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str, ckpt_dir: str, device: torch.device):
    ckpt_path = Path(ckpt_dir)
    is_lora = (ckpt_path / "adapter_config.json").exists()

    if is_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("加载 LoRA adapter 需要 peft 库：pip install peft>=0.14.0")
        print(f"检测到 LoRA adapter，加载 base model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(Path(model_path).resolve()), trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            str(Path(model_path).resolve()),
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
        print(f"加载 LoRA adapter: {ckpt_dir}")
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        model = model.merge_and_unload()
    else:
        print(f"检测到全量微调 checkpoint，直接加载: {ckpt_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(ckpt_path), trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path),
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )

    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    ckpt_type = "LoRA adapter 已合并" if is_lora else "全量微调模型"
    print(f"模型加载完成（{ckpt_type}）\n")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 推理与解析
# ══════════════════════════════════════════════════════════════════════════════

def classify_pair(s1: str, s2: str, model, tokenizer,
                  device: torch.device, max_new_tokens: int = 8) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"句子A：{s1}\n句子B：{s2}\n是否相似："},
    ]
    encoding = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    prompt_len = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_prediction(raw_output: str) -> int:
    """
    解析模型输出：
      含【不相似】→ 0  （需先检查，避免【相似】被误匹配）
      含【相似】  → 1
      其他        → -1（parse_fail）
    """
    if "不相似" in raw_output:
        return 0
    if "相似" in raw_output:
        return 1
    return -1


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="LLM SFT 文本匹配评估")
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument("--ckpt_dir", default=str(ADAPTER_DIR),
                        help="checkpoint 目录；含 adapter_config.json → LoRA，否则 → 全量")
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--num_samples", default=200, type=int,
                        help="验证集采样数")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--demo", action="store_true",
                        help="只跑 5 条示例，快速演示")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        print(f"[错误] checkpoint 目录不存在：{ckpt_dir}")
        print("请先运行 train_sft.py 完成训练。")
        print("  LoRA（默认）保存到:   outputs/sft_adapter/")
        print("  全量微调保存到:        outputs/sft_full_ckpt/")
        return

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    data_path = Path(args.data_dir) / "validation.jsonl"
    val_data = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]

    random.seed(args.seed)
    n = 5 if args.demo else args.num_samples
    samples = random.sample(val_data, min(n, len(val_data)))
    print(f"评估样本数: {len(samples)}\n")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model_path, str(ckpt_dir), device)

    # ── 推理 ──────────────────────────────────────────────────────────────────
    gold_labels, pred_labels = [], []
    parse_fail = 0
    detail_records = []
    t0 = time.time()

    for i, item in enumerate(samples, 1):
        s1, s2 = item["sentence1"], item["sentence2"]
        gold = item["label"]
        raw = classify_pair(s1, s2, model, tokenizer, device)
        pred = parse_prediction(raw)

        if pred == -1:
            parse_fail += 1
        else:
            gold_labels.append(gold)
            pred_labels.append(pred)

        detail_records.append({
            "sentence1": s1, "sentence2": s2,
            "label": gold, "pred": pred, "raw_output": raw,
        })

        gold_str = "相似" if gold == 1 else "不相似"
        pred_str = "相似" if pred == 1 else ("不相似" if pred == 0 else "解析失败")
        status = "✓" if pred == gold else ("?" if pred == -1 else "✗")
        print(f"[{i:3d}/{len(samples)}] {status}  真实:{gold_str}  预测:{pred_str}"
              f"  |  {s1[:20]} / {s2[:20]}")

    elapsed = time.time() - t0

    # ── 计算指标 ──────────────────────────────────────────────────────────────
    if gold_labels:
        acc = accuracy_score(gold_labels, pred_labels)
        f1 = f1_score(gold_labels, pred_labels, average="weighted", zero_division=0)
        f1_pos = f1_score(gold_labels, pred_labels, average="binary", zero_division=0)
    else:
        acc = f1 = f1_pos = 0.0

    # ── 读取已有结果做多方对比 ─────────────────────────────────────────────────
    def read_log_f1(path, key="f1"):
        if Path(path).exists():
            with open(path, encoding="utf-8") as file:
                data = json.load(file)
            # 训练日志是 list，取最后一条；评估结果是 dict
            if isinstance(data, list):
                return data[-1].get("val_f1", data[-1].get("f1", "?"))
            return data.get(key, data.get("metrics", {}).get(key, "?"))
        return "?"

    bi_cosine_f1 = read_log_f1(LOG_DIR / "biencoder_cosine_log.json")
    bi_triplet_f1 = read_log_f1(LOG_DIR / "biencoder_triplet_log.json")
    cross_f1 = read_log_f1(LOG_DIR / "crossencoder_log.json")

    llm_api_acc = "?"
    llm_api_f1 = "?"
    llm_log = LOG_DIR / "llm_compare_results.json"
    if llm_log.exists():
        with open(llm_log, encoding="utf-8") as f:
            llm_data = json.load(f)
        llm_api_acc = f"{llm_data['metrics'].get('accuracy', '?'):.4f}"
        llm_api_f1 = f"{llm_data['metrics'].get('f1_pos', '?'):.4f}"

    # 辅助函数：格式化 F1（可能是 float 或字符串）
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    print(f"\n{'=' * 65}")
    print(f"LLM SFT 文本匹配评估结果")
    print(f"{'=' * 65}")
    print(f"  样本数      : {len(samples)}（有效: {len(gold_labels)}，parse_fail: {parse_fail}）")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  F1 (weighted): {f1:.4f}")
    print(f"  F1 (正例)    : {f1_pos:.4f}")
    print(f"  均值耗时     : {elapsed / len(samples):.2f}s/条（GPU）")

    print(f"""
多方对比（AFQMC validation 集，所有方案均使用 Accuracy + F1，直接可比）
  ┌──────────────────────────────────────────┬──────────┬──────────┐
  │ 方法                                     │ Accuracy │ F1(pos)  │
  ├──────────────────────────────────────────┼──────────┼──────────┤
  │ BiEncoder + CosineEmbeddingLoss          │ （见日志）│ {fmt(bi_cosine_f1):<8} │
  │ BiEncoder + TripletLoss                  │ （见日志）│ {fmt(bi_triplet_f1):<8} │
  │ CrossEncoder + CrossEntropyLoss          │ （见日志）│ {fmt(cross_f1):<8} │
  │ Qwen API zero-shot                       │ {llm_api_acc:<8} │ {llm_api_f1:<8} │
  │ Qwen2-0.5B SFT（LoRA）                   │ {acc:.4f}   │ {f1_pos:.4f}   │
  └──────────────────────────────────────────┴──────────┴──────────┘

思考题：
  1. SFT 的 Accuracy 与 BERT 方案相比如何？用了多少训练数据？
  2. 生成式方法 parse_fail 率有多高？与 NER 任务相比为什么更低？
  3. BERT BiEncoder 可以做向量检索，SFT 方法可以吗？各自适合什么场景？
  4. 文本匹配的 SFT TARGET 只有 2~3 个 token，与 NER（20~150 token）相比训练有什么不同？
""")

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    out_path = LOG_DIR / "sft_results.json"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": acc, "f1_weighted": f1, "f1_pos": f1_pos,
            "n_samples": len(samples), "parse_fail": parse_fail,
            "detail": detail_records,
        }, f, ensure_ascii=False, indent=2)
    print(f"结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
