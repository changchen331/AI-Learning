"""
main.py — fincli：A股年报检索 + 天气查询 统一命令行入口

把 src/ 后端能力封装成一条"看起来像 git/ls 那样"的真实命令，而不是
`python xxx.py ...`。通过 pyproject.toml 的 [project.scripts] 注册为
console_script，`pip install -e .` 后即可全局调用：

  fincli list-companies
  fincli search --query "营收和净利润" --stock-code 300750 --year 2023 --top-k 3
  fincli weather --city 宁德

不想安装也可直接跑：
  python mode_cli/cli/main.py search --query "营收" --stock-code 300750 --year 2023
  python -m mode_cli.cli.main weather --city 宁德

教学点：
  1. CLI 作为"工具实现层"，本质就是一个能跑的脚本——跟协议无关
  2. 用 pyproject + console_script 把脚本变成 PATH 上的真实命令，是 Python CLI 工具的标准发布方式
  3. 一个 fincli 含多个子命令（search/list-companies/weather），对应 git 的子命令设计

依赖：
  pip install faiss-cpu numpy openai httpx
  环境变量：DASHSCOPE_API_KEY（Embedding）
"""

import argparse
import sys
from pathlib import Path

# 让本脚本能 import 项目根的 src/（无论从哪个工作目录 / 是否安装）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from week_11_2026_07_12.Examples.function_call_mcp_cli.src.rag_backend import search_annual_report, list_companies
from week_11_2026_07_12.Examples.function_call_mcp_cli.src.weather_backend import get_weather


def main():
    parser = argparse.ArgumentParser(
        prog="fincli",
        description="fincli — A股年报检索 + 天气查询 命令行工具",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # fincli search ...
    p_search = sub.add_parser("search", help="检索年报段落")
    p_search.add_argument("--query", required=True,
                          help="检索问题（不要含公司名/年份，用简短财务术语，如 '营收和净利润'）")
    p_search.add_argument("--stock-code", default=None, help="按公司过滤，如 300750")
    p_search.add_argument("--year", default=None, help="按年份过滤：2021/2022/2023")
    p_search.add_argument("--top-k", type=int, default=5, help="返回段落数，默认5")

    # fincli list-companies
    sub.add_parser("list-companies", help="列出知识库收录的公司")

    # fincli weather ...
    p_weather = sub.add_parser("weather", help="查询城市天气")
    p_weather.add_argument("--city", required=True, help="城市中文名，如 宁德")

    args = parser.parse_args()

    if args.cmd == "search":
        print(search_annual_report(args.query, args.stock_code, args.year, args.top_k))
    elif args.cmd == "list-companies":
        print(list_companies())
    elif args.cmd == "weather":
        print(get_weather(args.city))


if __name__ == "__main__":
    main()
