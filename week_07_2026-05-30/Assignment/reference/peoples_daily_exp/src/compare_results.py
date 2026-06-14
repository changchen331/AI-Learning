"""
人民日报 NER：四方案汇总对比脚本

使用方式：
  python compare_results.py
"""

import json
from pathlib import Path

EXP_ROOT = Path(__file__).parent.parent
LOG_DIR = EXP_ROOT / "outputs" / "logs"


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    linear_res = load_json(LOG_DIR / "eval_linear_validation.json")
    crf_res = load_json(LOG_DIR / "eval_crf_validation.json")
    linear_test = load_json(LOG_DIR / "eval_linear_test.json")
    crf_test = load_json(LOG_DIR / "eval_crf_test.json")
    llm_res = load_json(LOG_DIR / "eval_llm.json")
    sft_res = load_json(LOG_DIR / "eval_sft.json")

    print("\n" + "=" * 80)
    print("人民日报 NER —— 四方案汇总对比")
    print("=" * 80)

    print(f"\n{'方案':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'非法序列':>10}")
    print("-" * 80)

    if linear_res:
        ill = linear_res["illegal_stats"]["total_illegal"]
        print(f"{'BERT + Linear (验证集)':<35} "
              f"{linear_res['precision']:>10.4f} {linear_res['recall']:>10.4f} "
              f"{linear_res['f1']:>10.4f} {ill:>10d}")
    if crf_res:
        ill = crf_res["illegal_stats"]["total_illegal"]
        print(f"{'BERT + CRF (验证集)':<35} "
              f"{crf_res['precision']:>10.4f} {crf_res['recall']:>10.4f} "
              f"{crf_res['f1']:>10.4f} {ill:>10d}")
    if linear_test:
        ill = linear_test["illegal_stats"]["total_illegal"]
        print(f"{'BERT + Linear (测试集)':<35} "
              f"{linear_test['precision']:>10.4f} {linear_test['recall']:>10.4f} "
              f"{linear_test['f1']:>10.4f} {ill:>10d}")
    if crf_test:
        ill = crf_test["illegal_stats"]["total_illegal"]
        print(f"{'BERT + CRF (测试集)':<35} "
              f"{crf_test['precision']:>10.4f} {crf_test['recall']:>10.4f} "
              f"{crf_test['f1']:>10.4f} {ill:>10d}")
    if llm_res:
        zs = llm_res["zero_shot"]
        fs = llm_res["few_shot"]
        model_name = llm_res.get("model", "qwen-plus")
        n = llm_res.get("n_samples", "?")
        print(f"{f'LLM zero-shot ({model_name})':<35} "
              f"{zs['precision']:>10.4f} {zs['recall']:>10.4f} "
              f"{zs['f1']:>10.4f} {'N/A':>10}")
        print(f"{f'LLM few-shot ({model_name})':<35} "
              f"{fs['precision']:>10.4f} {fs['recall']:>10.4f} "
              f"{fs['f1']:>10.4f} {'N/A':>10}")
        print(f"  注：LLM 结果基于验证集 {n} 条采样")
    if sft_res:
        m = sft_res["metrics"]
        n = sft_res.get("n_samples", "?")
        print(f"{f'Qwen2-0.5B SFT (LoRA, {n} 条)':<35} "
              f"{m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {'N/A':>10}")

    print("=" * 80)


if __name__ == "__main__":
    main()
