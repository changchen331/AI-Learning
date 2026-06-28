"""
演示 vLLM 的 guided_json 约束解码：输出严格符合指定 JSON Schema

教学重点：
  1. guided_json 比 guided_choice/guided_regex 更强：一次约束整个复杂对象
  2. 对比 response_format={"type":"json_object"}：后者只保证"是 JSON"，不保证字段
  3. 这是 function call / tool call 场景的基础

场景：财报问答意图的结构化抽取
  用户问题 → {公司名, 年度, 指标} 三元组

使用方式：
  python demo_guided_json.py
"""

import json
import time

from jsonschema import validate, ValidationError
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
MODEL = "qwen2-0.5b"

# ── JSON Schema 定义 ──────────────────────────────────────────────────
INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "company": {
            "type": "string",
            "description": "公司全称，如 招商银行、贵州茅台",
        },
        "year": {
            "type": "integer",
            "minimum": 2015,
            "maximum": 2025,
        },
        "metric": {
            "type": "string",
            "enum": ["营收", "净利润", "ROE", "毛利率", "总资产", "经营现金流"],
        },
    },
    "required": ["company", "year", "metric"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = f"""你是财报问答助手。从用户问题中提取结构化信息，输出纯 JSON，不要任何解释文字。

字段定义：
  company: 公司全称
  year: 年度（2015~2025 整数）
  metric: 指标，必须是 ['营收', '净利润', 'ROE', '毛利率', '总资产', '经营现金流'] 之一

示例输出：
{{"company": "招商银行", "year": 2023, "metric": "营收"}}"""

TEST_CASES = [
    "招行 2023 年营收多少",
    "贵州茅台 2022 的净利润",
    "平安银行去年（2024）的 ROE",
    "2021 年五粮液毛利率",
    "2023 宁德时代经营现金流",
    "问一下比亚迪 2024 的总资产规模",
    "茅台 2020 年利润情况",  # "利润"不是枚举值，模型要映射到"净利润"
    "ICBC 2023 营收",  # 英文简称，测模型理解
    "隆基绿能 22 年 roe",  # 简写年份+小写指标
]


def evaluate(output: str) -> dict:
    """评估一个输出：分层校验（JSON 合法 / 字段齐全 / schema 完全通过）"""
    result = {
        "is_json": False,
        "has_all_fields": False,
        "year_in_range": False,  # 新增：year 必须 2015~2025
        "metric_in_enum": False,
        "schema_valid": False,  # 整体 jsonschema 校验
        "parsed": None,
    }
    try:
        obj = json.loads(output)
        result["is_json"] = True
        result["parsed"] = obj
    except json.JSONDecodeError:
        return result

    required = INTENT_SCHEMA["required"]
    if all(k in obj for k in required):
        result["has_all_fields"] = True

    yr = obj.get("year")
    if isinstance(yr, int) and 2015 <= yr <= 2025:
        result["year_in_range"] = True

    if obj.get("metric") in INTENT_SCHEMA["properties"]["metric"]["enum"]:
        result["metric_in_enum"] = True

    try:
        validate(instance=obj, schema=INTENT_SCHEMA)
        result["schema_valid"] = True
    except ValidationError:
        pass

    return result


def run_generate(user_msg: str, mode: str) -> tuple[str, float]:
    """mode: 'raw' | 'guided_json' | 'response_format'"""
    extra = {}
    kwargs = {}
    if mode == "guided_json":
        extra = {"guided_json": INTENT_SCHEMA}
    elif mode == "response_format":
        kwargs = {"response_format": {"type": "json_object"}}

    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=120,
        extra_body=extra,
        **kwargs,
    )
    return resp.choices[0].message.content.strip(), time.time() - t0


def main():
    print("=" * 78)
    print("  Demo: guided_json（JSON Schema 约束）")
    print(f"  Model: {MODEL}")
    print("  对比三种模式：裸 prompt / response_format / guided_json")
    print("=" * 78)

    counters = {m: {"json": 0, "fields": 0, "year": 0, "enum": 0, "valid": 0}
                for m in ["raw", "response_format", "guided_json"]}
    n = len(TEST_CASES)

    for user in TEST_CASES:
        print(f"\n▶ {user}")
        for mode in ["raw", "response_format", "guided_json"]:
            out, _ = run_generate(user, mode)
            ev = evaluate(out)
            c = counters[mode]
            if ev["is_json"]:         c["json"] += 1
            if ev["has_all_fields"]:  c["fields"] += 1
            if ev["year_in_range"]:   c["year"] += 1
            if ev["metric_in_enum"]:  c["enum"] += 1
            if ev["schema_valid"]:    c["valid"] += 1
            tag = "✓" if ev["schema_valid"] else "✗"
            disp = out[:80] + "…" if len(out) > 80 else out
            print(f"  [{mode:<16}] {tag}  {disp}")

    print("\n" + "=" * 78)
    print(f"  {n} 条测试结果汇总")
    print("=" * 78)
    print(f"{'指标':<24}{'裸 prompt':<18}{'response_format':<20}{'guided_json':<15}")
    print("-" * 78)
    for metric_name, key in [("合法 JSON", "json"),
                             ("字段齐全", "fields"),
                             ("year 在 2015~2025", "year"),
                             ("metric 在枚举内", "enum"),
                             ("jsonschema 完全通过", "valid")]:
        row = f"{metric_name:<22}"
        for mode in ["raw", "response_format", "guided_json"]:
            v = counters[mode][key]
            row += f"{v}/{n} ({100 * v / n:.0f}%)      "
        print(row)

    print()
    print("=" * 78)
    print("  结论：")
    print("    response_format 只保证是 JSON，不保证字段名、类型、枚举正确")
    print("    guided_json     是唯一 100% 保证 schema 合法的方式")
    print("=" * 78)


if __name__ == "__main__":
    main()
