"""
weather_server.py — 天气查询 MCP Server（方式二：MCP）

教学重点：
  1. 把 src/weather_backend 的同步函数包成 MCP 工具，加一行装饰器即可
  2. 与 rag_server 共存于不同子进程，由 Host 统一管理——展示 MCP"多 Server 聚合"

使用方式（由 run_mcp.py 作为子进程启动，stdio 通信）：
  python mode_mcp/servers/weather_server.py

依赖：
  pip install mcp httpx
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# 用 as 别名避免同名 tool 函数遮蔽后端函数导致递归
from week_11_2026_07_12.Examples.function_call_mcp_cli.src.weather_backend import \
    get_weather as _get_weather  # noqa: E402


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


mcp = FastMCP("weather-server")


@mcp.tool()
def get_weather(city: str) -> str:
    """
    查询指定城市的当前天气及未来3天预报。

    Args:
        city: 城市中文名，如 '宁德'、'北京'。同名地名会自动取行政级别更高的（如福建宁德而非西藏宁德）。

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述。
    """
    return _get_weather(city)


if __name__ == "__main__":
    log("Weather MCP Server 启动中（stdio 模式）...")
    mcp.run(transport="stdio")
