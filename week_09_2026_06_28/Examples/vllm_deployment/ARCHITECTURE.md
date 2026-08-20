# ARCHITECTURE.md — vLLM 部署与约束解码教学项目

## 一、项目定位

本项目通过 **6 个独立可运行的脚本**，把 vLLM 从"一个推理框架"拆解成学生可以感性理解的具体能力：

1. **部署能力**：一条命令把 HuggingFace 模型变成 OpenAI 兼容 HTTP 服务
2. **性能能力**：PagedAttention + continuous batching 的定量收益
3. **约束解码能力**：guided_choice / guided_regex / guided_json / response_format 四种解码约束的工程价值

场景选型： **金融问答 / 电商下单**——和课程体系中 `rag_annual_report`、`react_financial_agent` 等项目呼应，学生能把 vLLM
的推理层和 RAG/Agent 的应用层串起来。

### 核心方案对比

| 部署方案                    | 优点                                 | 缺点                     | 本项目立场        |
|-----------------------------|--------------------------------------|--------------------------|-------------------|
| Transformers `.generate()`  | 原生、零依赖                         | 吞吐极差、不支持约束解码 | 作为性能 baseline |
| Transformers batching 手写  | 简单能跑                             | padding 浪费、无动态调度 | 作为性能 baseline |
| vLLM OpenAI server          | OpenAI 兼容、动态 batching、约束解码 | 只支持 Linux             | **教学主线**      |
| TGI / TensorRT-LLM / SGLang | 各有专长                             | 复杂或厂商绑定           | 课后扩展阅读      |

---

## 二、整体流水线

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │   Qwen2-0.5B-Instruct (pretrain_models/ 已有, 0.94 GB on GPU)        │
 └──────────────────────────────────────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
 [A] transformers            [B] transformers           [C] vLLM engine
     serial                      batch=8                   (paged attn +
                                                          continuous batching)
        │                          │                          │
        │                          │                          │
        ▼                          ▼                          ▼
    60 tok/s                   289 tok/s                  3394 tok/s
     0.8 QPS                    3.9 QPS                   48.6 QPS
                                                             │
                                                             ▼
                                                  启动 OpenAI 兼容 HTTP 服务
                                                  (src/start_server.sh)
                                                             │
    ┌────────────────────────────────────────────────────────┴────────────────┐
    ▼                    ▼                    ▼                    ▼          ▼
demo_guided_      demo_guided_        demo_guided_       demo_response    demo_function
 choice.py         regex.py            json.py           _format.py        _call.py
（枚举）          （正则）            （Schema）          （JSON mode）      ★（综合）

                                                                       各 50×3 对比
                                                                       outputs/*.json
                                                                       throughput_*.png
```

各脚本对应：

- `src/start_server.sh` — vLLM server 一键启动
- `src/bench_throughput.py` — 路线 A/B/C 对比，产出 `throughput_comparison.png`
- `src/demo_guided_choice.py` — 枚举约束
- `src/demo_guided_regex.py` — 正则约束
- `src/demo_guided_json.py` — JSON Schema 约束（基础）
- `src/demo_response_format.py` — OpenAI 标准 `response_format`
- `src/demo_function_call.py` — **核心**：get_stock_quote + create_order 双工具 × 50 测试 × 3 模式

---

## 三、各环节技术选型

### 3.1 基座模型：Qwen2-0.5B-Instruct

**选型原因**：

- 已在 `pretrain_models/` 中，复用不重新下载（符合 CLAUDE.md 规范）
- 0.5B 足够小，适合 8GB 显存的笔记本卡跑批对比
- 0.5B 的"笨"恰好是教学优势——它会犯真实的错误（字段拼错、枚举值不匹配、JSON 断尾），让 guided_json 的价值被放大展示
- Qwen2 tokenizer 对中文支持好，适合金融领域中文 prompt

**备选**：

- Qwen2.5-1.5B：若显存允许，分类准确率会明显提升
- Qwen2.5-3B-AWQ：最佳体验但 WSL2 下 AWQ 偶有编译坑，本项目不用

### 3.2 推理引擎：vLLM 0.9.2

**选型原因**（"为什么不用最新版"）：

- vLLM 0.20+ 的 torch 依赖是 CUDA 13 runtime，要求 NVIDIA 驱动 580+
- 常见的 RTX 3060/4060 笔记本用户驱动是 566.x（CUDA 12.7）
- 降到 vLLM 0.9.2 + torch 2.7+cu126，向下兼容全部现役 NVIDIA 驱动

**关键启动参数**（`start_server.sh`）：

| 参数                       | 值      | 理由                                                   |
|----------------------------|---------|--------------------------------------------------------|
| `--max-model-len`          | 2048    | 0.5B 模型不需要长上下文，省 KV cache                   |
| `--gpu-memory-utilization` | 0.6     | 留 40% 显存给并行的 transformers benchmark             |
| `--dtype`                  | float16 | Qwen2 半精度足够，bf16 也可                            |
| `--enforce-eager`          | 加      | 关 CUDA graph 首次启动快 5~10 秒（教学不追求极致性能） |
| `--host 0.0.0.0`           |         | 允许 Windows 浏览器访问                                |

### 3.3 约束解码：guided_choice / guided_regex / guided_json

vLLM 内部用 **xgrammar + outlines_core** 实现约束解码。原理：

1. 解码前把 Schema/正则/枚举构建成 FSM
2. 每步解码时，根据当前 FSM 状态把非法 token 的 logit 设为 `-inf`
3. Softmax 后只能从合法 token 集里采样

教学演示的"失败模式"精心设计（见 `demo_function_call.py`）：

- 字段名拼错：`stock_code` vs `symbol`
- 枚举值多余文本：`"credit card"` vs `"card"`
- 正则不符：`"+13812345678"` 前缀加号
- 数值范围溢出：`year: 22`（应 ≥ 2015）
- 必选字段缺失：漏 `market`、漏 `fields`

### 3.4 评估框架：jsonschema + 手写指标

选型原因：

- 标准 JSON Schema 生态，可移植
- 分层指标： **JSON 合法 → 字段齐全 → 完整 schema 通过**，让学生看清 3 种模式的差异到底在哪一层

---

## 四、实验结果

### 4.1 吞吐对比（Qwen2-0.5B / RTX 4060 Laptop 8GB / WSL2）

| 模式                                | 50 请求总耗时 | QPS       | Generation tok/s | 相对 vLLM |
|-------------------------------------|---------------|-----------|------------------|-----------|
| [A] transformers 串行               | **60.98s**    | 0.82      | 60               | 0.017×    |
| [B] transformers batch=8（padding） | 12.85s        | 3.89      | 289              | 0.080×    |
| [C] vLLM continuous batching        | **1.03s**     | **48.59** | **3394**         | **1.00×** |

**vLLM 相对 transformers 串行加速 59.3×；相对 batch=8 加速 12.5×。**

结果解读：

- **A → B** 加速 4.7×：简单批处理就能大幅提速，但 padding 到最长 prompt 仍有大量 token 浪费
- **B → C** 加速 12.5×：continuous batching + PagedAttention 才是真正质变：
    - 短请求完成后立即插入新请求，长请求不拖累短请求
    - KV cache 按 block 动态分配，消除 padding 空间浪费
    - batch size 可动态到 20~40，GPU 利用率接近 100%

### 4.2 约束解码效果（get_stock_quote，50 条测试）

| 指标                 | 裸 prompt | response_format | guided_json |
|----------------------|-----------|-----------------|-------------|
| JSON 语法合法        | 86%       | 100%            | 100%        |
| 必选字段齐全         | 86%       | 100%            | 100%        |
| **完整 schema 通过** | **60%**   | **68%**         | **100%**    |
| 平均延迟（秒）       | 0.43      | 0.41            | 0.43        |

### 4.3 约束解码效果（create_order，50 条测试）

| 指标                 | 裸 prompt | response_format | guided_json |
|----------------------|-----------|-----------------|-------------|
| JSON 语法合法        | 96%       | 100%            | 100%        |
| 必选字段齐全         | 96%       | 100%            | 100%        |
| **完整 schema 通过** | **42%**   | **42%**         | **100%**    |
| 平均延迟（秒）       | 0.57      | 0.56            | 0.58        |

**核心结论**：

- `response_format={"type":"json_object"}` 把 JSON 合法率从 ~90% 拉到 100%，但 **字段语义准确率不改善**
- `guided_json=schema` 是唯一能把完整 schema 通过率拉到 100% 的方式
- 约束解码几乎不增加延迟（FSM 一次构建长期复用）

### 4.4 典型失败案例（教学现场展示）

```
Prompt: "300750 宁德时代最高价"
裸 prompt / response_format:
  {"symbol": "300750", "market": "SH", "date": "2026-05-12",
   "fields": ["最高价"], "adjust": "none"}
  ← fields[0] 是中文"最高价"，不符合 enum ["open","close","high","low","volume"]
guided_json:
  {"symbol": "300750", "market": "SZ", "date": "2026-05-12",
   "fields": ["high"], "adjust": "none"}
  ← 强制映射到合法枚举

Prompt: "订 5 本《三体》，联系人 13711112222，快递"
裸 prompt / response_format:
  {..., "user_phone": "+13711112222", ...}
  ← 模型加了国际区号"+"，违反正则 ^1[3-9]\d{9}$
guided_json:
  {..., "user_phone": "13711112222", ...}
  ← FSM 屏蔽了"+"字符
```

---

## 五、消融实验建议（课堂/课后扩展）

若时间充裕，可演示的消融维度：

| 消融维度                                 | 关键观察                                                        |
|------------------------------------------|-----------------------------------------------------------------|
| `enforce_eager` on/off                   | CUDA Graph 优化约 +20% QPS，但首次启动慢 5~10s                  |
| `gpu_memory_utilization` 0.3 / 0.6 / 0.9 | KV cache 越大，batch concurrency 越高                           |
| 模型规模 0.5B / 1.5B / 3B                | tok/s 下降但 guided_json schema 通过率显著上升                  |
| prompt 长度变化                          | 短请求占 batch slot 少，长请求是 continuous batching 收益放大器 |

---

## 六、关键工程决策与踩坑

| 问题                                                 | 根因                                                                                    | 解法                                                           |
|------------------------------------------------------|-----------------------------------------------------------------------------------------|----------------------------------------------------------------|
| `torch.cuda.is_available()` 返回 False               | vLLM 0.20.x 依赖 torch 2.11 + CUDA 13，需要驱动 580+；常见笔记本驱动是 566（CUDA 12.7） | 降级 `pip install vllm==0.9.2`（带 torch 2.7+cu126）           |
| `aimv2 is already used by a Transformers config`     | transformers 5.x 内置 aimv2，与 vLLM 0.9.2 代码冲突                                     | `pip install transformers==4.52.4`                             |
| `ValueError: Using a device_map requires accelerate` | transformers `from_pretrained(device_map="cuda")` 依赖 accelerate                       | `pip install accelerate`                                       |
| 中文路径 `/mnt/d/badou/项目材料准备/...` 加载模型    | UTF-8 路径在 vLLM log 里看着乱码但实际能用                                              | 无需修复，log 里 `\uxxxx` 只是显示问题                         |
| 第一次 `wsl --install` 后 Ubuntu 不自动弹窗          | Win11 新版行为，只装了 WSL 子系统                                                       | 再次执行 `wsl --install -d Ubuntu-22.04` 会真正下载发行版      |
| `pkill -f 'vllm.entrypoints'` 把自己的 bash 也杀了   | `-f` 模糊匹配命令行，bash 命令里也有这字符串                                            | 改用 `fuser -k 8000/tcp` 按端口杀                              |
| matplotlib 显示"串行"方块                            | DejaVu Sans 不含中文字形                                                                | 图表标签改英文（`serial`/`batch`）                             |
| bench 同时跑 transformers + vLLM 显存不足            | 两份 0.5B 模型 + KV cache                                                               | 先停掉 vLLM server 再跑 bench                                  |
| WSL2 默认装在 C 盘，长期会挤满                       | 默认路径 `%LOCALAPPDATA%\wsl\`                                                          | `wsl --export` + `wsl --unregister` + `wsl --import` 迁到 D 盘 |

---

## 七、优化方向（对学生的开放话题）

**数据层面**

- 构造更贴近业务的 function call 测试集（从真实 agent 对话日志抽取）
- 加入多轮对话场景：`get_stock_quote` 后 `compare_stocks`

**模型层面**

- 换 Qwen2.5-1.5B / 3B-AWQ，观察 schema 通过率是否继续上升
- 用 SFT 微调 0.5B 做 function call 专用模型，对比 API 成本

**解码层面**

- `guided_grammar` EBNF 构造更复杂的命令解析（当前项目未涵盖）
- xgrammar 的 compile 缓存命中率观察

**工程层面**

- 接入 Prometheus 监控 `/metrics` 接口，画 Grafana 大盘
- Triton / TGI / SGLang 横向对比
- 集成到 MCP Server（复用 `mcp_financial_agent` 项目）

---

## 八、目录结构

```
vllm_deployment/
├── ARCHITECTURE.md              # 本文档：技术方案
├── USAGE_GUIDE.md               # 环境搭建 + 脚本使用
├── RESUME_GUIDE.md              # 求职简历指导
├── requirements.txt             # Python 依赖
│
├── src/
│   ├── start_server.sh          # 一键启动 vLLM server
│   ├── bench_throughput.py      # transformers vs vLLM 吞吐对比
│   ├── demo_guided_choice.py    # 枚举约束（意图路由）
│   ├── demo_guided_regex.py     # 正则约束（日期/股票代码）
│   ├── demo_guided_json.py      # JSON Schema 约束（财报三元组）
│   ├── demo_response_format.py  # OpenAI 标准 json_object
│   └── demo_function_call.py    # ★ 核心：双工具 × 50×3 对比
│
└── outputs/
    ├── throughput_comparison.png       # 吞吐柱状图
    ├── throughput_results.json         # 吞吐原始数据
    └── function_call_results.json      # 约束解码详细结果
```
