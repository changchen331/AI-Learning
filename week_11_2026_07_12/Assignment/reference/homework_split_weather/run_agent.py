"""
run_agent.py — 作业入口：两个拆分后的天气工具 + agent loop

教学重点：
  1. 两个独立工具 geocode / get_weather_by_coords（见 weather_tools.py）
  2. 【真正的 agent loop】——不是"一次 tool_call + 一次回填"的单轮，而是
     while 循环：模型输出 tool_call → 执行 → 回填 → 再问模型，直到模型不再
     调用工具、给出最终回答为止。
       - 问"宁德天气"：模型先调 geocode(宁德)→拿到经纬度→再调 get_weather_by_coords→回答（链式/循环）
       - 问"北京的经纬度"：模型只调一次 geocode 即可回答（单工具独立答）
       - 问"经度116.4 纬度39.9 的天气"：模型只调一次 get_weather_by_coords（单工具独立答）
     这三种形态共用同一个 loop，差异完全由模型自己决定——这就是 agent loop。

LLM 接口：参考项目里 mode_function_call/run_function_call.py 的 DeepSeek 调用
（OpenAI 兼容协议，openai SDK）。

使用：
  # 配置环境变量
  #   Windows:  set DEEPSEEK_API_KEY=sk-xxx
  #   Linux:    export DEEPSEEK_API_KEY=sk-xxx

  # 单个问题
  python homework_split_weather/run_agent.py -q "宁德总部今天的天气怎么样？"

  # 内置示例（演示链式/单工具三种形态）
  python homework_split_weather/run_agent.py --demo

  # 切到 DashScope 的 qwen-plus
  python homework_split_weather/run_agent.py --provider dashscope -q "北京的经纬度"
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# 让 weather_tools 可被 import（直接 python 运行本脚本也能找到）
sys.path.insert(0, str(Path(__file__).parent))

from weather_tools import geocode, get_weather_by_coords  # noqa: E402

# ── LLM 配置（参考 mode_function_call/run_function_call.py）─────────────────

PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "dashscope": {
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


# ── 工具 Schema：两个拆分后的工具 ───────────────────────────────────────────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "geocode",
            "description": (
                "把城市名解析成经纬度（地理编码）。输入中文城市名如'北京'、'宁德'，"
                "返回该城市的纬度 latitude 和经度 longitude。"
                "当用户问'某城市的经纬度/坐标'时直接用本工具即可；"
                "当用户问'某城市天气'但本工具不含天气查询时，先用本工具拿到经纬度，"
                "再把经纬度传给 get_weather_by_coords 查天气（链式调用）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'、'北京'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_by_coords",
            "description": (
                "按经纬度查询当前天气及未来3天预报。参数是数值型的纬度/经度。"
                "若用户已直接给出经纬度，直接调用本工具；"
                "若用户只给了城市名，请先调用 geocode 拿到经纬度，再调用本工具（链式）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "纬度，如 39.9"},
                    "longitude": {"type": "number", "description": "经度，如 116.4"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
]

# 工具名 → 后端函数 的派发表（业务逻辑与协议层分离）
TOOL_DISPATCH = {
    "geocode": geocode,
    "get_weather_by_coords": get_weather_by_coords,
}

SYSTEM_PROMPT = (
    "你是一名天气助手，有两个工具可用：geocode（城市名→经纬度）和 "
    "get_weather_by_coords（经纬度→天气）。"
    "请按需调用工具，必要时可链式调用（先 geocode 拿经纬度，再 get_weather_by_coords 查天气）。"
    "只依据工具返回的数据作答，不要编造。"
)

# ── 【核心】agent loop：模型循环调用工具直到给出最终回答 ────────────────────

MAX_STEPS = 10  # 防御性兜底，避免模型无限循环


def run(client, model: str, question: str, verbose: bool = True) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []

    for step in range(1, MAX_STEPS + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # 模型本轮不再调用工具 → 已经是最终回答，退出循环
        if not msg.tool_calls:
            if verbose:
                print(f"  → [llm] 最终回答（第{step}轮，共{time.time() - t0:.1f}s）")
            return {
                "answer": msg.content or "",
                "tool_calls": tool_call_log,
                "steps": step,
                "elapsed": time.time() - t0,
            }

        # 把 assistant 这条带 tool_calls 的消息原样回填，保持上下文
        messages.append(msg)

        # 逐个执行模型本轮要调的工具
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args})
            if verbose:
                print(f"  → [tool step {step}] {name}({args})")
            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = f"未知工具：{name}"
            else:
                try:
                    result = fn(**args)
                except TypeError as e:
                    result = f"参数错误：{e}"
                except Exception as e:
                    result = f"工具执行失败：{e}"
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
            # 以 role=tool 回填，tool_call_id 必须对上
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        # 循环回到顶部，让模型看到工具结果后决定：继续调工具 or 给最终回答

    return {
        "answer": "（达到最大步数，模型仍未给出最终回答）",
        "tool_calls": tool_call_log,
        "steps": MAX_STEPS,
        "elapsed": time.time() - t0,
    }


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "宁德今天的天气怎么样？",  # 链式：geocode → get_weather_by_coords
    "北京的经纬度是多少？",  # 单工具：只 geocode
    "经度116.4、纬度39.9 这个地方天气如何？",  # 单工具：只 get_weather_by_coords
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="作业：拆分天气工具 + agent loop")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    print(f"[Split Weather Agent] provider={args.provider} model={model}\n")

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    for i, q in enumerate(questions, 1):
        print("=" * 60)
        print(f"Q{i}：{q}")
        print("=" * 60)
        result = run(client, model, q, verbose=not args.quiet)
        print("\n最终回答：")
        print(result["answer"])
        print(f"\n（工具调用 {len(result['tool_calls'])} 次，循环 {result['steps']} 轮，"
              f"耗时 {result['elapsed']:.1f}s）\n")


if __name__ == "__main__":
    main()
