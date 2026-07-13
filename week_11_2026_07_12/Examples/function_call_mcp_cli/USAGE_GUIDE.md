# USAGE_GUIDE.md — 代码调用与测试指南

## 1. 环境准备

### 1.1 依赖安装

```bash
cd function_call_mcp_cli
pip install -r requirements.txt
```

依赖：`openai`、`faiss-cpu`、`numpy`、`httpx`、`mcp>=1.0.0`。无需 Node.js（本版不使用 Filesystem MCP）。

**把 fincli 装成真实命令（推荐，一次即可）**：

```bash
pip install -e .
```

这会把 `fincli` 注册到 PATH（`pyproject.toml` 的 `[project.scripts]`），之后在任意目录都能像 `git`/`ls` 一样直接敲
`fincli search ...`。方式三的 CLI 形态依赖它；不装也能用（自动退回 `python mode_cli/cli/main.py`，只是命令不"漂亮"）。

### 1.2 API Key 配置（系统环境变量）

本项目不使用 `.env` 文件，API Key 一律从系统环境变量读取。

Windows（PowerShell）：

```powershell
$env:DEEPSEEK_API_KEY  = "sk-xxx"   # 默认 LLM，驱动工具调用
$env:DASHSCOPE_API_KEY = "sk-xxx"   # Embedding 必需；备选 LLM qwen-plus
```

Windows（CMD，当前进程）：

```cmd
set DEEPSEEK_API_KEY=sk-xxx
set DASHSCOPE_API_KEY=sk-xxx
```

Linux / macOS：

```bash
export DEEPSEEK_API_KEY=sk-xxx
export DASHSCOPE_API_KEY=sk-xxx
```

- DeepSeek 申请：https://platform.deepseek.com/
- DashScope 申请：https://dashscope.aliyun.com/
- 想永久生效：Windows 用 `setx DEEPSEEK_API_KEY "sk-xxx"`（重开终端生效）；Linux/macOS 写入 `~/.bashrc` 或 `~/.zshrc`。

### 1.3 复制向量索引

```bash
python scripts/copy_data.py --src ../rag_annual_report/vectorstore -y
```

会把 `faiss_index.bin`（~40MB）和 `faiss_meta.json`（~15MB）复制到 `vectorstore/`。索引含 5 家公司 × 3 年共 15 份年报、10353
个分块。

---

## 2. 各方式运行

> 三种方式的命令行接口统一：`--question/-q` 单问题、`--demo` 内置 4 问、`--provider deepseek|dashscope`、`--json --quiet`（供
> compare.py）。

### 2.1 方式一：Function Call

```bash
python mode_function_call/run_function_call.py -q "宁德时代2023年营收和净利润？"
python mode_function_call/run_function_call.py --demo
python mode_function_call/run_function_call.py -q "..." --provider dashscope
```

**内部流程**：

1. 手写 `TOOLS_SCHEMA`（3 个工具的 JSON Schema）
2. `chat.completions.create(tools=TOOLS_SCHEMA, tool_choice="auto")`
3. 若返回 `tool_calls`：查 `TOOL_DISPATCH` 表调后端函数 → 结果以 `role=tool` 回填
4. 再次 `create`，模型生成最终回答

**预期输出**：先打印 `→ [tool] search_annual_report({...})` 与结果预览，再打印最终回答（含营收 4009 亿、归母净利润 441
亿等数据）。

### 2.2 方式二：MCP

```bash
python mode_mcp/run_mcp.py -q "宁德时代2023年营收和净利润？另外总部宁德的天气如何？"
python mode_mcp/run_mcp.py --demo
```

**内部流程**：

1. `AsyncExitStack` 顺序启动 rag_server / weather_server 两个子进程（stdio）
2. 每个 Server `initialize()` 握手 → `list_tools()` 发现工具，打印 `✓ [rag] search_annual_report, list_companies`；同时把
   MCP `inputSchema` 转 OpenAI `parameters`（一次走完连接+发现+转 schema）
3. LLM 单轮闭环：`tool_call` → 查 `tool_registry` 路由到对应 `ClientSession` → `session.call_tool()` → 结果回填
4. 模型生成最终回答

**预期输出**：stderr 打印连接日志 + `共 3 个工具就绪`；stdout 打印工具调用与最终回答（含年报数据 + 宁德天气）。

> MCP Server 的 INFO 日志会打到 stderr（如 `Processing request of type CallToolRequest`），属正常协议日志。

### 2.3 方式三：CLI

CLI 方式的核心思想是"把能力做成普通命令行工具"——它本身跟大模型没有任何关系，可以像 `ls`、`git` 一样独立使用；然后再让大模型通过一个
`run_cli`/`run_bash` 工具去调用它。所以下面分两步看：先把它当普通 CLI 用，再把它接给模型。

#### 2.3.1 作为命令行工具单独使用（不经过 LLM）

`mode_cli/cli/main.py` 是统一入口，`pip install -e .` 后就是 PATH 上的 `fincli` 命令。任何人都能直接敲命令拿到结果，跟大模型没有任何关系（RAG
检索仍需 `DASHSCOPE_API_KEY` 做 Embedding）：

```bash
# 列出知识库收录的公司
fincli list-companies

# 检索年报段落（query 不含公司名/年份，用简短财务术语）
fincli search --query "营收和净利润" --stock-code 300750 --year 2023 --top-k 3

# 查天气
fincli weather --city 宁德
```

> 没装 fincli？用 `python mode_cli/cli/main.py search ...` 或 `python -m mode_cli.cli.main weather --city 宁德` 也能跑，效果一样。

`fincli` 就是把 `src/` 后端函数包了一层 `argparse`，输出走 stdout 纯文本。这就是 CLI 作为"工具实现层"的全部——*
*一条能跑、能管道拼接的真实命令，无需任何协议**。可以配合 shell 管道用：

```bash
fincli search --query "营收" --stock-code 600519 --year 2023 | head -20
```

#### 2.3.2 结合大模型调用

`run_cli.py` 把上面的 `fincli` 包成 LLM 可调用的工具，单轮闭环。有两种形态，用 `--mode` 切换：

**形态 A（具名 `run_cli`，白名单，默认，更安全）**：

```bash
python mode_cli/run_cli.py --mode named -q "宁德时代2023年营收和净利润？"
```

LLM 调 `run_cli(command='rag_search', args={...})`，host 按 `NAMED_COMMANDS` 白名单拼出 `fincli search ...` 执行。
`command` 是 enum，模型只能选预批准的命令——安全可控，但每加一个命令要改代码。

**形态 B（通用 `run_bash`，沙箱，更灵活）**：

```bash
python mode_cli/run_cli.py --mode bash -q "宁德时代2023年营收和净利润？另外总部宁德的天气？"
```

LLM 自己拼完整 shell 命令字符串（如 `fincli search --query '营收和净利润' --stock-code 300750 --year 2023`），host 经
`sandbox_check` 后 `subprocess.run(shell=True)` 执行。最灵活也最危险，靠沙箱兜底。

> 两种形态对比是本方式的教学重点：形态 A = 安全的"工具白名单"，形态 B = 灵活的"通用 shell"，差异只在沙箱设计。

#### 2.3.3 沙箱拦截验证（形态 B）

```bash
python -c "from mode_cli.run_cli import run_bash; print(run_bash('rm -rf /'))"
# → [run_bash] 沙箱拦截：命中危险模式 '\\brm\\b'
```

沙箱 = 危险命令正则黑名单（rm/del/format/sudo/curl|sh/nc…）+ 命令头白名单（fincli/python/git/ls/cat/echo/type/dir）+ 15s
超时 + 工作目录锁定。

---

## 3. 三个方式对比

```bash
python compare.py
python compare.py --questions "宁德时代2023年营收？" "比亚迪2023年营收？"
python compare.py --provider dashscope
```

对每个问题依次跑 Function Call / MCP / CLI(named) / CLI(bash) 四方式，记录工具调用、耗时、是否拒绝幻觉，输出对比表到
`output/compare_result.md`，同时在控制台打印简表。

**预期**：四方式对同一问题调用工具基本一致；Function Call 进程内直调最快，MCP/CLI 有子进程或 IPC 开销更高；问比亚迪（不在库）时四方式都正确拒绝。

---

## 4. 作为模块调用

```python
import sys

sys.path.insert(0, ".")
from src.rag_backend import search_annual_report, list_companies
from src.weather_backend import get_weather

# 直接调后端
print(search_annual_report("营收和净利润", stock_code="300750", year="2023", top_k=3))
print(get_weather("宁德"))
```

```python
# 方式一作为模块
from mode_function_call.run_function_call import build_client, run, TOOLS_SCHEMA

client, model = build_client("deepseek")
result = run(client, model, "宁德时代2023年营收？", verbose=True)
print(result["answer"], result["tool_calls"])
```

---

## 5. 调试与常见问题

**Q1：`RecursionError: maximum recursion depth exceeded`（MCP 方式）**
MCP Server 的 tool 函数与导入的后端函数同名导致递归。本项目已用 `as` 别名修复，若你新增 Server 工具，注意别让 tool 函数名与
import 的后端函数名相同。

**Q2：MCP 启动报 `未设置 DASHSCOPE_API_KEY`**
rag_server 启动时需 Embedding。确认系统环境变量 `DASHSCOPE_API_KEY` 已设置（`echo $env:DASHSCOPE_API_KEY` 能看到值）。MCP
Server 子进程通过 `env={**os.environ}` 继承父进程环境变量。

**Q3：`vectorstore/faiss_index.bin 不存在`**
先跑 `python scripts/copy_data.py --src ../rag_annual_report/vectorstore -y`。

**Q4：DeepSeek 不返回 tool_calls，直接回答了**
偶发现象。系统 prompt 已强调"必须先调 search_annual_report"。若仍频繁出现，换 `--provider dashscope`（qwen-plus）。

**Q5：CLI(bash) 模型生成的命令被沙箱拦截**
看拦截信息：命中黑名单（危险命令）或命令头不在白名单（只允许 python/git/ls/cat/echo/type/dir）。模型应在 prompt 提示的命令集内生成。

**Q6：天气查到奇怪的地方（如"宁德"查到西藏）**
已修复：geocoding 自动取行政级别更高的候选，并对裸城市名追加"市"重查。若仍异常，显式传 `--city 宁德市`。

**Q7：Windows 报 `OMP: Error #15`**
`rag_backend.py` 顶部已设 `KMP_DUPLICATE_LIB_OK=TRUE`。若从外部 import 仍报，在你的入口脚本顶部同样设置。

**Q8：`python -m py_compile` 提示文件名以数字开头无法 import**
本项目所有文件用 snake_case，无数字前缀，可直接 import。
