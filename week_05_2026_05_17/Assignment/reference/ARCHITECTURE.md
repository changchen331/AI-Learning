# ARCHITECTURE.md — LLM 预训练教学项目技术方案

## 一、项目定位

**教学目标**：让学生亲手实现一个完整的大语言模型预训练流程，理解自回归语言模型的训练目标、架构设计、PPL 评估和多种解码策略。

**核心技术栈**：

- 语料：中文 Wikipedia（50k 篇子集，约 150MB 原始文本）
- Tokenizer：bert-base-chinese（字级 BPE，vocab=21128，已有本地文件）
- 模型：Mini GPT（~18.8M 参数，Decoder-only Transformer）
- 训练：原生 PyTorch 训练循环，AdamW + 余弦 LR + Gradient Clipping
- 评估：验证集 PPL（困惑度）
- 生成：Greedy / Temperature / Top-K / Top-P 四种解码策略

---

## 二、整体流水线

```
原始中文 Wikipedia
       │
       ▼
[download_data.py]
  下载前 50k 篇文章
  存为 data/wiki_zh.jsonl
       │
       ▼
[prepare_dataset.py]
  BertTokenizer 逐文章 tokenize
  拼接成连续 token 流（文章间插入 [SEP]）
  按 seq_len=256 切块
  → data/train_seq256.pt（N×257 张量）
  → data/val_seq256.pt（M×257 张量）
       │
       ▼
[train.py]
  加载 TokenDataset
  构建 MiniGPT（18.8M 参数）
  AdamW + CosineAnnealingLR
  自回归 Loss（CrossEntropy，input=t[:-1], target=t[1:]）
  每步计算 grad_norm，clip=1.0
  每 epoch 评估 val PPL，保存 checkpoint
  → outputs/checkpoints/best_model.pt
  → outputs/training_log.jsonl
       │
       ├──────────────────────────┐
       ▼                          ▼
[evaluate.py]               [generate.py]
  精确 PPL（token 级）          四种解码策略对比
  绘制训练曲线                  Greedy / Temp / Top-K / Top-P
  → outputs/training_curve.png  生成续写文本
```

---

## 三、模型架构详解

### 3.1 整体结构（Decoder-only GPT）

```
输入 token_ids (B, T)
      │
      ├─ Token Embedding  (V=21128, d=384)
      ├─ Position Embedding (max_len=256, d=384)
      │
      ▼
6 × TransformerBlock
  ├─ LayerNorm（Pre-LN，在残差支路之前）
  ├─ CausalSelfAttention（n_heads=6, d_head=64）
  │    ├─ QKV 投影（一个大矩阵，再 split）
  │    ├─ 缩放点积注意力 QK^T / sqrt(64)
  │    ├─ 因果掩码（上三角 = -inf）
  │    └─ 输出投影
  ├─ LayerNorm
  └─ FeedForward (d_model=384 → d_ff=1536 → d_model=384)
       └─ GELU 激活
      │
      ▼
Final LayerNorm
      │
      ▼
LM Head（Linear, 384 → 21128，权重与 Token Embedding 共享）
      │
      ▼
logits (B, T, 21128)
```

### 3.2 参数规模分布

| 组件                    | 参数量                     | 占比 |
|-------------------------|----------------------------|------|
| Token Embedding         | 21128 × 384 = 8.1M         | 43%  |
| Position Embedding      | 256 × 384 = 0.1M           | <1%  |
| 6 × Attention (QKV+Out) | 6 × 4 × 384² = 3.5M        | 19%  |
| 6 × FFN (两层线性)      | 6 × 2 × 384×1536 = 7.1M    | 38%  |
| LM Head                 | 共享 Embedding，0 额外参数 | 0%   |
| **总计**                | **~18.8M**                 |      |

### 3.3 关键设计选择原因

| 设计         | 选择                         | 原因                                           |
|--------------|------------------------------|------------------------------------------------|
| 位置编码     | 可学习 Embedding             | 实现简单，便于学生理解；sin/cos 留作扩展练习   |
| 归一化位置   | Pre-LN（LN 在残差前）        | 训练更稳定，不需要仔细调 warmup；现代 GPT 标准 |
| 激活函数     | GELU                         | GPT-2/3 标准选择，比 ReLU 在 LM 任务上略优     |
| Weight Tying | LM Head 共享 Token Embedding | 减少 8M 参数，经验上提升生成质量               |
| Tokenizer    | bert-base-chinese            | 本地已有，字级别直观，vocab 覆盖常见汉字       |

---

## 四、训练配置

| 超参数       | 值              | 说明                              |
|--------------|-----------------|-----------------------------------|
| seq_len      | 256             | 上下文窗口长度                    |
| batch_size   | 32              | 8G 显存舒适上限                   |
| lr           | 3e-4            | AdamW 典型学习率                  |
| weight_decay | 0.1             | 只施加在 2D+ 参数（线性层权重）上 |
| grad_clip    | 1.0             | 防止梯度爆炸                      |
| betas        | (0.9, 0.95)     | GPT-3 论文推荐                    |
| scheduler    | CosineAnnealing | lr 从 3e-4 余弦衰减到 3e-5        |
| epochs       | 3               | 快速验证；完整训练可跑 5~10       |

**显存估算（8G GPU）**：

- 模型权重：18.8M × 4 bytes = 75MB
- Optimizer states（AdamW 存 m/v）：×3 = 225MB
- 激活值（batch=32, seq=256, d=384）：~1GB
- **总计 < 2GB，8G 显存非常充裕**

---

## 五、PPL 评估

**公式**：

```
PPL = exp(H)
H   = -1/N × Σ log P(token_i | token_0..i-1)
    = avg CrossEntropy Loss（token 级均值）
```

**评估说明**：

- 必须用 **总 token 数**做分母，不能用 batch 数（每 batch token 数相同时等价，但语义更准确）
- 验证集 PPL < 训练集 PPL 说明过拟合
- 中文 Wikipedia 语料，~25M 参数模型，3 epoch 后 val PPL 预期降至 **50~150** 区间 （随机猜测基线：PPL ≈ 21128）

---

## 六、解码策略对比

| 策略        | 参数         | 确定性   | 文本质量               | 多样性 |
|-------------|--------------|----------|------------------------|--------|
| Greedy      | —            | 完全确定 | 容易重复/陷入循环      | 最低   |
| Temperature | T=0.8        | 随机     | 比 greedy 更流畅       | 中     |
| Top-K       | K=50, T=0.8  | 随机     | 截断长尾噪声           | 中高   |
| Top-P       | p=0.9, T=0.8 | 随机     | 自适应候选数量，最稳定 | 高     |

**教学要点**：Top-K 的缺陷是固定候选数——分布尖锐时 K=50 仍可能包含大量噪声，分布平坦时 K=50 又太少；Top-P 通过累积概率自适应解决此问题。

---

## 七、优化方向

### 数据层面

- 增大语料规模到全量 wiki_zh（~100万篇）
- 混入更多领域语料（新闻、书籍、代码）
- 数据清洗：去重、过滤低质量文章

### 模型层面

- 增加层数（6→12）、扩大 d_model（384→768）到 GPT-2 small 规模
- 引入 Rotary Position Embedding（RoPE）替换可学习位置编码
- 引入 Flash Attention 加速（`F.scaled_dot_product_attention`）

### 训练策略

- 增加 warmup steps（前 1% 步骤线性升温）
- 混合精度训练（`torch.autocast`），显存减半、速度翻倍
- 梯度累积（小 GPU 模拟大 batch）

### 工程部署

- 量化推理（INT8）
- KV Cache 加速生成
- 模型转 ONNX 格式

---

## 八、关键工程决策与踩坑

| 问题                                 | 根因                                                                          | 解法                                                                                                      |
|--------------------------------------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Windows OpenMP 冲突                  | torch 与 numpy 各自链接 libiomp5md.dll                                        | 所有脚本顶部加 `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")`                                    |
| transformers 不接受 Windows 相对路径 | 含 `..` 的路径被当 HuggingFace repo ID                                        | 用 `Path(__file__).parent.parent` 构造绝对路径                                                            |
| HuggingFace XET 存储国内无法访问     | hf-mirror.com 将大文件 302 到 `cas-bridge.xethub.hf.co`（美国 CDN），国内封锁 | 彻底绕开 HF：用 `requests` 直连 `dumps.wikimedia.org` 官方 dump，流式 BZ2 解压 + `XMLPullParser` 增量解析 |
| `datasets 4.0` 不支持 dataset script | 旧 `wikipedia` 仓库报 `RuntimeError: Dataset scripts are no longer supported` | 同上，直接用 Wikimedia dump，不经 datasets 库                                                             |
| Top-P 实现细节                       | 累积概率超过 p 后需"保留第一个超过的 token"                                   | `sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()`，index 0 强制保留                  |
| Weight Tying 后 LM Head 梯度重复     | `lm_head.weight = token_emb.weight` 共享同一 tensor                           | PyTorch 自动合并梯度，无需额外处理                                                                        |

---

## 九、目录结构

```
llm_pretrain/
├── src/
│   ├── download_data.py      # 下载中文 Wikipedia 子集 → data/wiki_zh.jsonl
│   ├── prepare_dataset.py    # tokenize + 拼接切块 → data/train/val_seq256.pt
│   ├── model.py              # MiniGPT 模型定义（Attention/FFN/TransformerBlock）
│   ├── train.py              # 训练主循环（含 PPL 评估、checkpoint）
│   ├── evaluate.py           # 精确 PPL 评估 + 训练曲线绘制
│   └── generate.py           # 四种解码策略文本生成
│
├── data/
│   ├── wiki_zh.jsonl         # 原始文章（~150MB，git ignore）
│   ├── train_seq256.pt       # 训练集张量（~400k 样本）
│   └── val_seq256.pt         # 验证集张量（~20k 样本）
│
├── outputs/
│   ├── checkpoints/
│   │   ├── epoch1_ppl***.pt  # 各 epoch checkpoint
│   │   └── best_model.pt     # 最优验证集 PPL 模型
│   ├── training_log.jsonl    # 每 epoch loss/PPL 记录
│   └── training_curve.png    # 训练曲线图（evaluate.py --plot 生成）
│
├── requirements.txt
├── ARCHITECTURE.md           # 本文件
├── USAGE_GUIDE.md
└── RESUME_GUIDE.md
```
