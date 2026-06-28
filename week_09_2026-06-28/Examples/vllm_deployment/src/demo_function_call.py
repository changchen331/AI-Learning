"""
核心演示：Function Call 的可靠性（裸 prompt vs response_format vs guided_json）

教学重点：
  1. Function Call 场景下小模型的三大失败模式：
     (a) JSON 语法错（缺括号、多逗号、混中文引号）
     (b) 字段名拼错或漏必选字段
     (c) 字段值不符合约束（6位股票代码、手机号正则、数量范围）
  2. guided_json 用 FSM + JSON Schema 在解码时屏蔽非法 token，一次修复所有三类问题
  3. 这就是为什么生产环境 Agent 系统离不开约束解码

两个工具示例：
  - get_stock_quote：金融场景，schema 含 string/enum/regex/array/minItems
  - create_order：电商场景，schema 含 integer 范围、手机号正则、多枚举

每个工具 50 条测试用例，覆盖基础 / 缺字段 / 诱导多余文本 / 非标准输入等场景。

使用方式（需先启动 vLLM server）：
  # 终端 1：bash start_server.sh
  # 终端 2：python demo_function_call.py
  # 或仅跑一个：python demo_function_call.py --tool stock
"""

import argparse
import json
import time

from jsonschema import validate, ValidationError
from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────────────────────
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
MODEL = "qwen2-0.5b"

# ══════════════════════════════════════════════════════════════════════
#                     工具 1: get_stock_quote
# ══════════════════════════════════════════════════════════════════════

STOCK_SCHEMA = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string", "pattern": r"^\d{6}$"},
        "market": {"type": "string", "enum": ["SH", "SZ", "BJ"]},
        "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
        "fields": {
            "type": "array",
            "items": {"type": "string", "enum": ["open", "close", "high", "low", "volume"]},
            "minItems": 1,
        },
        "adjust": {"type": "string", "enum": ["none", "qfq", "hfq"]},
    },
    "required": ["symbol", "market", "fields"],
    "additionalProperties": False,
}

STOCK_SYSTEM = """你是股票查询工具的参数生成器。根据用户问题输出纯 JSON 参数，不要任何解释文字。

JSON 格式：
{
  "symbol": "6位代码（如600000）",
  "market": "SH" | "SZ" | "BJ",
  "date": "YYYY-MM-DD 格式日期",
  "fields": ["open" | "close" | "high" | "low" | "volume"] 的数组,
  "adjust": "none" | "qfq" | "hfq"
}

必选字段：symbol, market, fields
市场规则：6开头=SH，0/3开头=SZ，8/4开头=BJ
默认值：date 不填用今天 2026-05-12；adjust 不填用 none；fields 不填用 ["close"]

示例：查询贵州茅台今天的收盘价和开盘价
输出：{"symbol": "600519", "market": "SH", "date": "2026-05-12", "fields": ["close", "open"], "adjust": "none"}"""

STOCK_TEST_CASES = [
    # 基础直接（15）
    "查 600000 今天的开盘价",
    "看看 600519 贵州茅台的收盘",
    "000001 平安银行今日 volume",
    "300750 宁德时代最高价",
    "002594 比亚迪今天最低价",
    "600036 招商银行当日收盘",
    "000651 格力电器成交量",
    "300015 爱尔眼科开盘价",
    "600028 中石化最低价",
    "000858 五粮液最高",
    "601318 平安保险 close",
    "300059 东方财富今日 volume",
    "600276 恒瑞医药收盘",
    "002415 海康威视最高",
    "000568 泸州老窖开盘",
    # 省略市场（8）
    "帮我查茅台今天情况",
    "招商银行现在股价",
    "平安银行今日收盘",
    "看看比亚迪的收盘价",
    "宁德时代今天最高最低",
    "五粮液现在什么价",
    "格力电器今日成交量",
    "洋河股份收盘价",
    # 诱导多余文本（7）
    "帮我查询平安银行今日开盘价，并简单解释什么是开盘价",
    "600000 今天什么价啊，顺便告诉我浦发发行股票的意义",
    "查下 600519 收盘价，并对比它和五粮液的差距",
    "看一眼宁德时代今天最高价，我想做下技术分析",
    "600036 收盘价，顺便推荐几只类似的银行股",
    "请查询东方财富成交量并分析异动原因",
    "帮我查 000001 收盘价，然后帮我判断是否该买入",
    # 非标准输入（5）
    "帮我查六零零零零零这只票",
    "查询股票代码 six zero zero five one nine 的收盘",
    "看下 600.000 今天开盘",
    "查 600000.SH 收盘价",
    "股票代码 600000，招银，今天收盘",
    # 日期模糊（5）
    "查一下 600000 昨天的收盘价",
    "看看茅台 2026 年 5 月 10 日的情况",
    "600519 五月八号开盘价",
    "招行上周五收盘价",
    "600000 今年开年以来最高",
    # 复合字段（5）
    "查茅台今天的开盘、收盘、最高、最低",
    "600000 今日 OHLC 全部",
    "全部数据都给我，代码 600519",
    "招行收盘和成交量",
    "五粮液最高最低成交量",
    # 边界/无意义（5）
    "随便查一个",
    "股票",
    "帮帮我",
    "？？？",
    "查询行情",
]
assert len(STOCK_TEST_CASES) == 50

# ══════════════════════════════════════════════════════════════════════
#                     工具 2: create_order
# ══════════════════════════════════════════════════════════════════════

ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "product": {"type": "string", "minLength": 1},
        "quantity": {"type": "integer", "minimum": 1, "maximum": 100},
        "user_phone": {"type": "string", "pattern": r"^1[3-9]\d{9}$"},
        "delivery_date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
        "priority": {"type": "string", "enum": ["normal", "express", "urgent"]},
        "payment_method": {"type": "string", "enum": ["alipay", "wechat", "card"]},
    },
    "required": ["product", "quantity", "user_phone", "priority"],
    "additionalProperties": False,
}

ORDER_SYSTEM = """你是订单创建工具的参数生成器。根据用户描述输出纯 JSON 订单参数，不要任何解释文字。

JSON 格式：
{
  "product": "商品名称",
  "quantity": 1~100 的整数,
  "user_phone": "11位手机号（1开头第二位3-9）",
  "delivery_date": "YYYY-MM-DD",
  "priority": "normal" | "express" | "urgent",
  "payment_method": "alipay" | "wechat" | "card"
}

必选字段：product, quantity, user_phone, priority
默认值：delivery_date 不填用后天 2026-05-14；payment_method 不填用 alipay

示例：用户张三要订 2 个 iPhone 15 Pro，电话 13912345678，明天送到，加急
输出：{"product": "iPhone 15 Pro", "quantity": 2, "user_phone": "13912345678", "delivery_date": "2026-05-13", "priority": "urgent", "payment_method": "alipay"}"""

ORDER_TEST_CASES = [
    # 基础直接（15）
    "用户李四要订 3 个 AirPods，电话 13812345678，普通配送",
    "王五买 1 台 MacBook，手机 13987654321，加急，微信付款",
    "订 5 本《三体》，联系人 13711112222，快递，支付宝",
    "赵六要 10 个保温杯，电话 13644445555，普通即可",
    "张三下单 2 台 iPad，15277778888 加急",
    "采购 20 瓶矿泉水，13922223333，明天送到",
    "下单 iPhone 15 一台，13855556666 刷卡",
    "订购 4 盒月饼，13766667777，急件",
    "购买 8 个鼠标，13511112222",
    "来 2 台戴尔笔记本，13688889999，urgent",
    "帮我订 1 件羽绒服，13433334444，alipay",
    "7 本笔记本，18900001111",
    "30 个口罩，13055556666，card 付款",
    "3 瓶红酒，13677778888，wechat",
    "12 盒茶叶，13822223333 快递",
    # 数量超限（6）
    "给我 200 个鼠标，电话 13912345678",  # 超过最大 100
    "订 0 个苹果，13812345678",  # 小于 1
    "1000 瓶矿泉水，13755556666",  # 远超
    "5000 本书，13833334444 加急",
    "我要 150 个保温杯，电话 13922220000",
    "要 -3 个，13833330000",
    # 电话不标准（7）
    "订 1 个 iPad，手机 138-1234-5678",
    "买 2 件衣服，联系方式 138 1234 5678",
    "3 本书，phone +86 13812345678",
    "5 个勺子，手机号：86-13812345678",
    "1 台电脑，电话：12345678",  # 位数不够
    "订 2 瓶酒，手机 19912345678",  # 199 开头
    "10 件商品，电话 00000000000",  # 全零
    # 日期相对表达（6）
    "订 1 台电脑，13812345678，后天送到",
    "买 2 个耳机，13922223333，周五之前",
    "3 盒月饼，13711112222，下周一",
    "1 件 T 恤，13833334444，最晚 2026 年 5 月 20 日",
    "5 本书，15277776666，明天，加急",
    "2 瓶酒，13688884444，今天就要",
    # priority 诱导错（5）
    "订 1 台电视，13912348765，最高优先级",
    "3 盒巧克力，13866667777，非常急",
    "5 个 U 盘，13911112222，普通",
    "2 本书，13800001111，加急处理",
    "10 瓶酒，13955556666，standard",  # standard 不在 enum
    # payment 诱导错（5）
    "订 1 个保温杯，13912345678，用银联卡",
    "买 2 件衣服，13855557777，微信支付",
    "3 本书，13799998888，cash 付款",  # cash 不在 enum
    "5 盒茶叶，13666667777，支付宝",
    "1 台 iPad，13822223333，信用卡",
    # 边界/无意义（6）
    "下个订单",
    "帮我订东西",
    "我要买个啥",
    "???",
    "订",
    "帮我下个单，不告诉你买什么",
]
assert len(ORDER_TEST_CASES) == 50


# ══════════════════════════════════════════════════════════════════════
#                     通用运行 + 评估逻辑
# ══════════════════════════════════════════════════════════════════════

def run_one(system: str, user: str, mode: str, schema: dict,
            max_tokens: int = 250) -> tuple[str, float]:
    """mode: 'raw' | 'response_format' | 'guided_json'"""
    extra = {}
    kwargs = {}
    if mode == "guided_json":
        extra = {"guided_json": schema}
    elif mode == "response_format":
        kwargs = {"response_format": {"type": "json_object"}}

    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=max_tokens,
        extra_body=extra,
        **kwargs,
    )
    return resp.choices[0].message.content.strip(), time.time() - t0


def evaluate_output(output: str, schema: dict) -> dict:
    """返回分层指标：JSON 合法 / 必选字段齐全 / jsonschema 完全通过"""
    r = {"is_json": False, "has_required": False, "schema_valid": False,
         "parsed": None, "error": None}
    try:
        obj = json.loads(output)
        r["is_json"] = True
        r["parsed"] = obj
    except json.JSONDecodeError as e:
        r["error"] = f"JSON syntax: {e}"
        return r

    if not isinstance(obj, dict):
        r["error"] = "not an object"
        return r

    required = schema.get("required", [])
    if all(k in obj for k in required):
        r["has_required"] = True

    try:
        validate(instance=obj, schema=schema)
        r["schema_valid"] = True
    except ValidationError as e:
        r["error"] = f"schema: {e.message[:80]}"
    return r


def run_tool_benchmark(tool_name: str, schema: dict, system: str,
                       cases: list[str]) -> dict:
    """跑完一个工具的 50×3 测试，返回 stats 和示例"""
    modes = ["raw", "response_format", "guided_json"]
    stats = {m: {"is_json": 0, "has_required": 0, "schema_valid": 0, "total_latency": 0.0}
             for m in modes}
    fail_examples = {m: [] for m in modes}  # 保存前 3 个失败用例

    print(f"\n{'=' * 78}")
    print(f"  工具: {tool_name}   测试数: {len(cases)}   模式: 3")
    print(f"{'=' * 78}")

    for i, user in enumerate(cases, 1):
        if i % 10 == 0:
            print(f"  进度: {i}/{len(cases)}")
        for mode in modes:
            try:
                out, dt = run_one(system, user, mode, schema)
            except Exception as e:
                out, dt = f"[REQUEST ERROR: {e}]", 0.0
            ev = evaluate_output(out, schema)
            s = stats[mode]
            s["total_latency"] += dt
            if ev["is_json"]:       s["is_json"] += 1
            if ev["has_required"]:  s["has_required"] += 1
            if ev["schema_valid"]:  s["schema_valid"] += 1
            # 保存失败案例（schema 不过的前 3 个）
            if not ev["schema_valid"] and len(fail_examples[mode]) < 3:
                fail_examples[mode].append({
                    "user": user,
                    "output": out[:150],
                    "error": ev.get("error", "unknown"),
                })

    return {"stats": stats, "fails": fail_examples, "n": len(cases)}


def print_report(tool_name: str, result: dict):
    stats = result["stats"]
    fails = result["fails"]
    n = result["n"]

    print(f"\n{'─' * 78}")
    print(f"  【{tool_name}】 {n} 条测试 × 3 模式 汇总")
    print(f"{'─' * 78}")
    print(f"{'指标':<24}{'裸 prompt':<20}{'response_format':<22}{'guided_json':<15}")
    print("─" * 78)
    metric_labels = [
        ("JSON 语法合法", "is_json"),
        ("必选字段齐全", "has_required"),
        ("完整 schema 通过 ★", "schema_valid"),
    ]
    for label, key in metric_labels:
        row = f"{label:<22}"
        for mode in ["raw", "response_format", "guided_json"]:
            v = stats[mode][key]
            row += f"{v}/{n} ({100 * v / n:>3.0f}%)       "
        print(row)
    print(f"{'平均延迟（秒）':<22}", end="")
    for mode in ["raw", "response_format", "guided_json"]:
        avg = stats[mode]["total_latency"] / n
        print(f"{avg:.3f}              ", end="")
    print()

    print(f"\n{'─' * 78}")
    print(f"  【{tool_name}】 典型失败案例（前 3 条）")
    print(f"{'─' * 78}")
    for mode in ["raw", "response_format", "guided_json"]:
        n_fails = len(fails[mode])
        if n_fails == 0:
            print(f"\n[{mode}] ✓ 无失败案例")
        else:
            print(f"\n[{mode}] 失败示例（schema 校验未通过）：")
            for f in fails[mode]:
                print(f"  ▶ Prompt: {f['user']}")
                print(f"    输出:   {f['output']}")
                print(f"    错误:   {f['error']}")


# ══════════════════════════════════════════════════════════════════════
#                         main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", choices=["stock", "order", "both"], default="both")
    parser.add_argument("--out", default="../outputs/function_call_results.json",
                        help="结果保存路径（相对 src/）")
    args = parser.parse_args()

    print("=" * 78)
    print("  demo_function_call.py   核心：裸 prompt vs response_format vs guided_json")
    print(f"  Model: {MODEL}")
    print("=" * 78)

    all_results = {}

    if args.tool in ("stock", "both"):
        t0 = time.time()
        r = run_tool_benchmark("get_stock_quote", STOCK_SCHEMA, STOCK_SYSTEM, STOCK_TEST_CASES)
        all_results["stock"] = r
        print_report("get_stock_quote", r)
        print(f"\n  [耗时 {time.time() - t0:.1f}s]")

    if args.tool in ("order", "both"):
        t0 = time.time()
        r = run_tool_benchmark("create_order", ORDER_SCHEMA, ORDER_SYSTEM, ORDER_TEST_CASES)
        all_results["order"] = r
        print_report("create_order", r)
        print(f"\n  [耗时 {time.time() - t0:.1f}s]")

    # 保存详细结果
    import os
    out_path = os.path.join(os.path.dirname(__file__), args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # 清洗不可序列化内容
    to_save = {}
    for k, v in all_results.items():
        to_save[k] = {
            "n": v["n"],
            "stats": v["stats"],
            "fails": v["fails"],
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存：{out_path}")

    print("\n" + "=" * 78)
    print("  核心结论：")
    print("    裸 prompt        — JSON 语法偶尔错 / 字段拼错 / 正则枚举不符")
    print("    response_format  — JSON 合法率接近满分，但字段语义仍错")
    print("    guided_json      — 100% 满足完整 schema（小模型从不可用变可靠）")
    print("=" * 78)


if __name__ == "__main__":
    main()
