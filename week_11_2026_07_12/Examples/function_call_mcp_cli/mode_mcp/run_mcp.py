"""
run_mcp.py — 方式二：MCP Host（连接多 Server，单轮闭环调用）

教学重点：
  1. 工具来自"协议发现"而非手写：connect_all_servers 一次走完
     stdio_client 建管道 → initialize() 握手 → list_tools() 发现工具
  2. MCP 工具描述要转成 LLM 能懂的 OpenAI tools schema（inputSchema → parameters）
  3. 单轮闭环代码和 run_function_call.py 几乎一样——差异只在"工具从哪来/怎么执行"
     · Function Call：手写 schema + 直接调后端函数
     · MCP：发现 schema + 通过 call_tool 跨进程调用 Server
  4. AsyncExitStack 统一管理多个 Server 子进程的生命周期

使用方式：
  python mode_mcp/run_mcp.py --question "宁德时代2023年营收和净利润？"
  python mode_mcp/run_mcp.py --demo

依赖：
  pip install mcp openai
  环境变量：DEEPSEEK_API_KEY（默认 LLM）
            DASHSCOPE_API_KEY（Embedding，rag_server 内部用）

MCP 三角关系：
  Host（本文件）= 连接管理 + 工具路由 + LLM 单轮闭环
  Client        = ClientSession，每个 Server 一个会话
  Server        = rag_server.py / weather_server.py（子进程，stdio 通信）
"""

import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

BASE_DIR = Path(__file__).parent.parent

# ── LLM 配置（与 run_function_call.py 完全一致，便于横向对比）──────────────

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


# ── Server 配置 ────────────────────────────────────────────────────────────

def build_server_configs() -> dict[str, StdioServerParameters]:
    # 两个自写 Server，都用项目内 Python 脚本启动，stdio 通信
    servers = BASE_DIR / "mode_mcp" / "servers"
    return {
        "rag": StdioServerParameters(
            command=sys.executable,
            args=[str(servers / "rag_server.py")],
            env={**os.environ},
        ),
        "weather": StdioServerParameters(
            command=sys.executable,
            args=[str(servers / "weather_server.py")],
            env={**os.environ},
        ),
    }


# ── 连接所有 Server：一次走完 建管道→握手→发现工具→转 schema ───────────────

async def connect_all_servers(stack: AsyncExitStack):
    """
    连接所有 MCP Server，返回 (tool_registry, openai_tools)：
      tool_registry : tool_name → (ClientSession, server_label)，用于路由 call_tool
      openai_tools  : 转成 OpenAI tools schema 的列表，直接喂给 LLM
    """
    print("正在连接 MCP Servers...\n", file=sys.stderr)
    tool_registry: dict[str, tuple[ClientSession, str]] = {}
    openai_tools: list[dict] = []

    for label, params in build_server_configs().items():
        # stdio_client 建立进程间通信管道（子进程的 stdin/stdout）
        read, write = await stack.enter_async_context(stdio_client(params))
        session: ClientSession = await stack.enter_async_context(ClientSession(read, write))

        # initialize() = MCP 握手，协商协议版本和能力
        await session.initialize()

        # list_tools() = 工具发现；同时把 MCP inputSchema 适配成 OpenAI parameters
        # —— 这一步是"协议层 → 模型层"的转换：MCP 让工具与模型解耦，
        #   但喂给具体 LLM 时仍要变成它认识的格式（inputSchema 本就是 JSON Schema，直接塞）
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            tool_registry[tool.name] = (session, label)
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                },
            })
        print(f"  ✓ [{label}]  {', '.join(t.name for t in tools_result.tools)}", file=sys.stderr)

    print(f"\n共 {len(tool_registry)} 个工具就绪\n", file=sys.stderr)
    return tool_registry, openai_tools


# ── 单轮闭环 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名金融分析助手。回答用户关于A股年报的问题时，必须先调用 search_annual_report 工具检索年报原文，"
    "只依据工具返回的段落作答，不要编造数据。如果用户问的公司不在知识库"
    "（贵州茅台/五粮液/宁德时代/海康威视/中国平安），请明确告知不在库内，不要臆测。"
    "涉及天气时调用 get_weather。本回合你可以一次调用多个工具。"
)


async def run(client, model: str, question: str,
              tool_registry: dict, openai_tools: list[dict], verbose: bool = True) -> dict:
    """单轮闭环：提问 → 模型输出 tool_call → 路由到 Server 执行 → 回填 → 最终回答。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []

    # 第一次请求：带上 tools，让模型决定是否调用工具
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=openai_tools, tool_choice="auto",
    )
    msg = resp.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args})
            if verbose:
                print(f"  → [mcp] {name}({args})")

            # 查路由表找到对应 Server 的 ClientSession，跨进程调用
            session, label = tool_registry.get(name, (None, None))
            if session is None:
                result = f"未知工具：{name}"
            else:
                # call_tool() = MCP 协议的 tools/call 请求，工具在 Server 子进程内执行
                call_result = await session.call_tool(name, args)
                result = "\n".join(b.text for b in call_result.content if hasattr(b, "text"))

            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ [{label}] {preview}{'...' if len(result or '') > 120 else ''}\n")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        # 第二次请求：模型看到工具结果，生成最终回答
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=openai_tools, tool_choice="auto",
        )
        msg = resp.choices[0].message

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "宁德时代2023年营收和净利润是多少？",
    "宁德时代2023年营收和净利润是多少？另外总部宁德的天气如何？",
    "对比贵州茅台和五粮液2023年的营收。",
    "比亚迪2023年营收是多少？",
]


async def main_async(provider: str, question: str | None, demo: bool, verbose: bool, as_json: bool):
    client, model = build_client(provider)
    if not as_json:
        print(f"[MCP] provider={provider} model={model}\n", file=sys.stderr)

    async with AsyncExitStack() as stack:
        tool_registry, openai_tools = await connect_all_servers(stack)

        questions = DEMO_QUESTIONS if demo else ([question] if question else [DEMO_QUESTIONS[0]])
        results = []
        for i, q in enumerate(questions, 1):
            if not as_json:
                print("=" * 60)
                print(f"Q{i}：{q}")
                print("=" * 60)
            result = await run(client, model, q, tool_registry, openai_tools,
                               verbose=verbose and not as_json)
            result["question"] = q
            results.append(result)
            if not as_json:
                print("\n最终回答：")
                print(result["answer"])
                print()

        if as_json:
            print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式二：MCP")
    parser.add_argument("--question", "-q")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出（被 compare.py 调用时用）")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 compare.py 解析）")
    args = parser.parse_args()
    asyncio.run(main_async(args.provider, args.question, args.demo, verbose=not args.quiet, as_json=args.json))


if __name__ == "__main__":
    main()
