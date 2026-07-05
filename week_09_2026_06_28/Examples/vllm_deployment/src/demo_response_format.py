"""
演示 OpenAI 标准的 response_format 接口（可移植方案）

教学重点：
  1. response_format 是 OpenAI 官方 API 规范，vLLM/Azure/together.ai 都兼容
  2. 相比 guided_json（vLLM 私有扩展），它的优势是代码跨平台可移植
  3. 但它只有 {"type": "json_object"} 这种弱约束，字段/类型/枚举仍可能错

三种 JSON 输出方式的选型建议：
  ┌──────────────────────┬──────────┬──────────┬──────────────────────────┐
  │ 方式                  │ 合法率    │ 可移植性  │ 适用场景                  │
  ├──────────────────────┼──────────┼──────────┼──────────────────────────┤
  │ 裸 prompt 指令        │ 低(40-60%)│ 100%     │ 大模型 / 容错要求低        │
  │ response_format       │ 高(80-95%)│ 高        │ 多厂商切换 / 一般业务      │
  │ guided_json           │ 100%     │ 低(vLLM)  │ 小模型 / 严格下游解析      │
  └──────────────────────┴──────────┴──────────┴──────────────────────────┘

使用方式：
  python demo_response_format.py
"""

import json
import time

from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
MODEL = "qwen2-0.5b"

SYSTEM_PROMPT = """你是新闻情感分析助手。分析用户给的新闻标题，输出 JSON 格式：
{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": 0.0~1.0 的数值,
  "keywords": ["关键词1", "关键词2"]
}
不要输出任何其他文字。"""

TEST_CASES = [
    "茅台三季度营收创历史新高，净利润同比增长 15%",
    "比亚迪召回 10 万辆电动车，涉及电池安全问题",
    "央行维持 LPR 利率不变",
    "宁德时代与宝马签订长期供货协议",
    "平安保险高管被调查，股价下跌 8%",
]


def run(user: str, mode: str) -> tuple[str, float]:
    """mode: 'raw' | 'json_object'"""
    kwargs = {}
    if mode == "json_object":
        kwargs["response_format"] = {"type": "json_object"}
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=150,
        **kwargs,
    )
    return resp.choices[0].message.content.strip(), time.time() - t0


def evaluate(output: str) -> dict:
    r = {"is_json": False, "has_sentiment": False, "valid_sentiment": False,
         "has_confidence": False, "has_keywords": False}
    try:
        obj = json.loads(output)
        r["is_json"] = True
    except json.JSONDecodeError:
        return r
    if "sentiment" in obj:
        r["has_sentiment"] = True
        if obj["sentiment"] in ("positive", "negative", "neutral"):
            r["valid_sentiment"] = True
    if "confidence" in obj and isinstance(obj["confidence"], (int, float)):
        r["has_confidence"] = True
    if "keywords" in obj and isinstance(obj["keywords"], list):
        r["has_keywords"] = True
    return r


def main():
    print("=" * 75)
    print("  Demo: response_format（OpenAI 标准 JSON 模式）")
    print(f"  Model: {MODEL}")
    print("=" * 75)

    stats = {m: {"json": 0, "sentiment": 0, "valid_sent": 0, "conf": 0, "keywords": 0}
             for m in ["raw", "json_object"]}

    for news in TEST_CASES:
        print(f"\n▶ {news}")
        for mode in ["raw", "json_object"]:
            out, dt = run(news, mode)
            ev = evaluate(out)
            s = stats[mode]
            for k, e in [("json", "is_json"), ("sentiment", "has_sentiment"),
                         ("valid_sent", "valid_sentiment"),
                         ("conf", "has_confidence"), ("keywords", "has_keywords")]:
                if ev[e]:
                    s[k] += 1
            flag = "✓" if ev["is_json"] else "✗"
            disp = out[:100] + "…" if len(out) > 100 else out
            print(f"  [{mode:<12}] {flag} {disp}")

    n = len(TEST_CASES)
    print("\n" + "=" * 75)
    print(f"  {n} 条测试结果")
    print("=" * 75)
    print(f"{'指标':<22}{'裸 prompt':<20}{'response_format':<20}")
    print("-" * 60)
    for name, key in [("合法 JSON", "json"),
                      ("有 sentiment 字段", "sentiment"),
                      ("sentiment 值合法", "valid_sent"),
                      ("有 confidence 字段", "conf"),
                      ("有 keywords 字段", "keywords")]:
        row = f"{name:<20}"
        for mode in ["raw", "json_object"]:
            v = stats[mode][key]
            row += f"{v}/{n} ({100 * v / n:.0f}%)      "
        print(row)

    print()
    print("=" * 75)
    print("  观察：")
    print("    response_format 显著提升 JSON 合法率，但字段语义仍靠模型自觉")
    print("    若需严格字段 schema，请用 guided_json（见 demo_function_call.py）")
    print("=" * 75)


if __name__ == "__main__":
    main()
