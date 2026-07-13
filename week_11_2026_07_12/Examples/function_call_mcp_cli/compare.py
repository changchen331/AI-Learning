"""
compare.py — 三方式对比运行器（教学 centerpiece）

对同一组问题，依次跑 Function Call / MCP / CLI(named) / CLI(bash) 四种方式，
记录：调用了哪些工具、耗时、最终答案摘要、是否正确拒绝幻觉。
打印对比表 + 写 output/compare_result.md。

教学点：四种方式"达成同样效果"，但接入成本 / 安全 / 跨模型复用性差异明显。

使用方式：
  python compare.py                 # 跑内置 4 个问题
  python compare.py --questions "宁德时代2023年营收？" "比亚迪2023年营收？"
  python compare.py --provider dashscope

环境变量：DEEPSEEK_API_KEY（默认 LLM）/ DASHSCOPE_API_KEY（Embedding + 备选 LLM）

依赖：各方式脚本的依赖（无需额外安装）
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
PY = sys.executable

# 四种方式 = 子进程命令模板
MODES = [
    ("Function Call", [PY, str(BASE_DIR / "mode_function_call" / "run_function_call.py"), "--json", "--quiet"]),
    ("MCP", [PY, str(BASE_DIR / "mode_mcp" / "run_mcp.py"), "--json", "--quiet"]),
    ("CLI(named)", [PY, str(BASE_DIR / "mode_cli" / "run_cli.py"), "--mode", "named", "--json", "--quiet"]),
    ("CLI(bash)", [PY, str(BASE_DIR / "mode_cli" / "run_cli.py"), "--mode", "bash", "--json", "--quiet"]),
]

DEFAULT_QUESTIONS = [
    "宁德时代2023年营收和净利润是多少？",
    "宁德时代2023年营收和净利润是多少？另外总部宁德的天气如何？",
    "对比贵州茅台和五粮液2023年的营收。",
    "比亚迪2023年营收是多少？",  # 幻觉控制：比亚迪不在知识库
]

# 幻觉控制判定：答案里若出现"不在"/"未收录"/"无法"/"没有"等拒绝词，视为正确拒绝
REFUSE_PATTERNS = ["不在", "未收录", "无法", "没有收录", "不在库", "不在知识库", "未能", "查不到"]


def run_one(mode_cmd: list, question: str, provider: str) -> dict:
    """以子进程跑一种方式，返回解析后的结果 dict。"""
    cmd = mode_cmd + ["--provider", provider, "-q", question]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180,
            cwd=str(BASE_DIR), env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "超时(>180s)", "elapsed": time.time() - t0}

    wall = time.time() - t0
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr[-500:], "elapsed": wall}

    # --json 模式 stdout 最后一行是 JSON
    out = proc.stdout.strip().splitlines()
    if not out:
        return {"ok": False, "error": "无输出：" + proc.stderr[-300:], "elapsed": wall}
    try:
        data = json.loads(out[-1])
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"JSON 解析失败：{e}", "elapsed": wall}

    data["ok"] = True
    data["wall_elapsed"] = wall
    return data


def summarize(data: dict, question: str) -> dict:
    """从一种方式的结果里抽要对比字段。"""
    if not data.get("ok"):
        return {
            "tools": "-",
            "tool_count": 0,
            "llm_elapsed": "-",
            "answer_preview": "(失败) " + (data.get("error", "")[:60]),
            "refused": False,
        }
    tcs = data.get("tool_calls", [])
    tool_names = ", ".join(t["name"] for t in tcs) or "(无工具调用)"
    answer = data.get("answer", "")
    # 幻觉拒绝判定：问题涉及"比亚迪"（不在库）且答案含拒绝词
    need_refuse = "比亚迪" in question
    refused = any(p in answer for p in REFUSE_PATTERNS) if need_refuse else None
    return {
        "tools": tool_names,
        "tool_count": len(tcs),
        "llm_elapsed": f"{data.get('elapsed', 0):.1f}s",
        "answer_preview": answer[:80].replace("\n", " ") + ("..." if len(answer) > 80 else ""),
        "refused": refused,
    }


def run_compare(questions: list[str], provider: str) -> list[dict]:
    rows = []
    for qi, q in enumerate(questions, 1):
        print(f"\n{'=' * 70}\nQ{qi}：{q}\n{'=' * 70}")
        for mode_name, mode_cmd in MODES:
            print(f"  ▶ {mode_name} ...", end=" ", flush=True)
            data = run_one(mode_cmd, q, provider)
            s = summarize(data, q)
            status = "✓" if data.get("ok") else "✗"
            print(f"{status} 工具[{s['tool_count']}] {s['llm_elapsed']}")
            rows.append({
                "question": q, "mode": mode_name,
                **s, "raw_ok": data.get("ok", False),
            })
    return rows


def write_markdown(rows: list[dict], questions: list[str], provider: str, path: Path):
    lines = [
        "# 三方式对比结果（Function Call / MCP / CLI）",
        "",
        f"- LLM provider：`{provider}`",
        f"- 生成时间：本表由 `python compare.py` 实跑生成",
        f"- 问题数：{len(questions)}，方式数：{len(MODES)}",
        "",
        "## 对比表",
        "",
        "| 问题 | 方式 | 工具调用 | 工具数 | LLM耗时 | 正确拒绝幻觉 | 答案摘要 |",
        "|------|------|---------|:------:|:-------:|:------------:|---------|",
    ]
    for r in rows:
        refuse_cell = "-" if r["refused"] is None else ("✓ 拒绝" if r["refused"] else "✗ 未拒绝(可能幻觉)")
        lines.append(
            f"| {r['question']} | {r['mode']} | {r['tools']} | {r['tool_count']} | "
            f"{r['llm_elapsed']} | {refuse_cell} | {r['answer_preview']} |"
        )

    lines += [
        "",
        "## 解读",
        "",
        "- **工具调用一致性**：四种方式对同一问题调用的工具与参数基本一致——"
        "说明底层能力相同，差异在『接入方式』而非『能力』。",
        "- **接入成本**：Function Call 要手写 schema；MCP 要写 Server 但工具自动发现可跨产品复用；"
        "CLI(named) 写白名单；CLI(bash) 几乎零封装但需沙箱。",
        "- **安全**：Function Call / MCP / CLI(named) 都走白名单，安全；CLI(bash) 依赖沙箱拦截，最危险。",
        "- **跨模型复用**：MCP 工具可被任意支持 MCP 的 Host 复用；Function Call schema 各家 API 略有差异；"
        "CLI 与模型完全无关。",
        "- **幻觉控制**：问比亚迪（不在知识库）时，看各方式是否正确拒绝而非编造数据。",
        "",
        "## 各方式原始回答",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="三方式对比运行器")
    parser.add_argument("--questions", nargs="+", default=DEFAULT_QUESTIONS)
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "dashscope"])
    args = parser.parse_args()

    print(f"[compare] provider={args.provider}, {len(args.questions)} 个问题 × {len(MODES)} 种方式\n")

    rows = run_compare(args.questions, args.provider)

    out_dir = BASE_DIR / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "compare_result.md"
    write_markdown(rows, args.questions, args.provider, out_path)

    # 控制台简表
    print(f"\n{'=' * 70}\n对比表（已写入 {out_path}）\n{'=' * 70}")
    print(f"{'问题':<28}{'方式':<16}{'工具数':<6}{'LLM耗时':<10}{'拒绝':<8}")
    print("-" * 70)
    for r in rows:
        refuse = "-" if r["refused"] is None else ("✓" if r["refused"] else "✗")
        print(f"{r['question'][:26]:<28}{r['mode']:<16}{r['tool_count']:<6}{r['llm_elapsed']:<10}{refuse:<8}")


if __name__ == "__main__":
    main()
