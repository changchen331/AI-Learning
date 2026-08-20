# ARCHITECTURE.md — 技术方案文档

## 1. 项目定位

本项目以 **中文命名实体识别（NER）** 为任务，核心教学目标是量化对比四套方案：

| 方案                 | 实现方式                            | 教学价值                             |
|----------------------|-------------------------------------|--------------------------------------|
| BERT + Linear        | 线性分类头，逐 token 独立预测       | BERT 序列建模基线，展示局限性        |
| BERT + CRF           | 条件随机场，全局序列解码            | 量化 CRF 对序列合法性的保证          |
| LLM API（qwen-plus） | zero-shot + few-shot 提示           | 揭示微调小模型 vs 大模型提示的差距   |
| LLM SFT（LoRA）      | 指令微调 Qwen2-0.5B，生成 JSON 实体 | 生成式 NER vs 序列标注；参数高效微调 |

**数据集**：cluener2020，10 类细粒度实体，来源 CLUE benchmark，训练集 10,748 条，验证集 1,343 条。

---

## 2. 整体流水线

```
原始数据（span格式）
    │
    ▼
src/download_data.py     ← 从 CLUE Google Storage 下载，无需 HuggingFace 登录
    │
    ▼
src/explore_data.py      ← 实体分布、文本长度、实体长度可视化
    │
    ▼
src/dataset.py           ← span → BIO 转换 + BERT 子词对齐
    │
    ▼
src/train.py             ← --use_crf 切换两种模型；AdamW 分层学习率 + linear warmup
    │
    ▼
src/evaluate.py          ← seqeval entity-level F1 + 非法序列统计

────── 并行 LLM 对比 ──────

src_llm/llm_ner.py       ← Qwen API zero-shot / few-shot（100条采样，分层抽样）
    │
    ▼
src_llm/train_sft.py     ← LoRA 指令微调（chat格式 + loss masking + --full_ft 开关）
    │
    ▼
src_llm/evaluate_sft.py  ← 加载 checkpoint（自动识别 LoRA/全量），span F1 四方对比

────── 汇总 ──────

src/compare_results.py   ← 汇总四套方案数字对比表
```

---

## 3. 各环节技术选型

### 3.1 数据处理：span 标注 → BIO 转换

**选型原因**：cluener2020 原始格式是 span（起止位置），BERT NER 的标准输入是 BIO。

转换流程（`dataset.py: span_to_bio()`）：

1. 初始化全 `O` 数组，长度等于文本字符数
2. 遍历 label 字典，按 `[start, end]` 填入 `B-X`（开头）和 `I-X`（续接）
3. 调用 BertTokenizer，`is_split_into_words=True`（逐字符输入）
4. 用 `word_ids()` 对齐子词标签：首子词保留 BIO，非首子词和特殊 token 设为 `-100`

**为什么要 -100**：PyTorch `cross_entropy(ignore_index=-100)` 和 CRF 的 mask 都支持跳过这些位置。

### 3.2 模型：BertNER vs BertCRFNER

**线性头（BertNER）**：

- `BertModel` → `Dropout` → `Linear(768, num_labels)` → token-level logits
- 每个 token 独立做 softmax，缺乏标签间约束
- **问题**：可能输出 `I-name` 开头、`B-name` 接 `I-company` 等非法序列

**CRF 头（BertCRFNER）**：

- 同上，但最后一层改为 `CRF(num_labels)`（来自 `pytorch-crf` 库）
- CRF 维护 `(num_labels × num_labels)` 的转移矩阵，学习哪些转移合法
- 训练：前向-后向算法计算对数似然；推理：Viterbi 找全局最优路径

**为什么 CRF 提升不多**：BERT 的双向注意力已能隐式建模上下文约束，CRF 的主要价值在于 **数学保证合法性**，而非大幅提升 F1。

### 3.3 训练：AdamW + 分层学习率

```python
{"params": bert_params, "lr": 2e-5}  # BERT 层：小步微调，避免破坏预训练参数
{"params": head_params, "lr": 1e-4}  # 分类头 + CRF：较大 lr，加速收敛
```

线性 warmup：总训练步的前 10% 从 0 升到目标 lr，之后线性衰减到 0。

### 3.4 统一评估体系

本项目四套方案使用两种相互兼容的评估方式， **最终数字可以在同一量纲下比较**。

**方式 A：seqeval entity-level F1**（BERT 方案，`evaluate.py`）

BERT NER 输出 BIO 标签序列，每个 token 对应精确位置，可直接构成 (text, type, start, end) 4 元组。seqeval 要求 entity
的类型和字符边界完全匹配才算 TP。

**方式 B：span F1 + text.find () 近似定位**（LLM 方案，`llm_ner.py` 和 `evaluate_sft.py`）

LLM 输出 JSON 格式实体列表，JSON 中不包含位置信息。通过 `text.find(surface)` 在原文中定位起始字符，构成 (text, type, start,
end) 4 元组，再与 gold 做集合交集计算 P/R/F1。`llm_ner.py` 和 `evaluate_sft.py` 使用完全相同的实现， **LLM API 和 SFT 的 F1
数字可直接比较**。

**两种方式的差异**：仅在同一句子里同一实体多次出现时，`text.find()` 只取第一个，位置可能与 gold 不一致。实测同一样本集：两种方式的
F1 差 < 0.01，差异可忽略，四套方案的 F1 基本在同一量纲下。

### 3.5 LLM API 对比：Qwen API + 结构化输出

- 模型：`qwen-plus`（DashScope 兼容接口）
- 输出格式：JSON `{"entities": [{"text": "...", "type": "..."}]}`
- 正则提取 JSON 块，容错处理非标准输出
- 评估：span F1 + text.find ()（方式 B）
- 成本控制：验证集采样 100 条（分层采样，保证 10 类实体均有覆盖）

### 3.6 LLM SFT：LoRA 指令微调

#### NER 的 chat 格式

```
system: 你是命名实体识别助手，以 JSON 格式输出...
        实体类型：address/book/company/.../scene
user:   浙商银行企业信贷部叶老桂博士则从另一个角度...
TARGET: {"entities": [{"text": "浙商银行", "type": "company"},
                      {"text": "叶老桂", "type": "name"}]}
```

**与分类任务的关键差异**：

| 维度            | 分类           | NER                     |
|-----------------|----------------|-------------------------|
| TARGET token 数 | 1~2（类别名）  | 20~150（JSON 实体列表） |
| loss 集中程度   | 极度集中       | 分散在整个 JSON         |
| 生成控制难度    | 低（单词约束） | 中（JSON 格式约束）     |
| max_length      | 128            | 256（JSON 输出更长）    |

#### Loss Masking

```python
prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
response_ids = tokenizer.encode(target_json, add_special_tokens=False) + [eos_id]
labels = [-100] * len(prompt_ids) + response_ids  # prompt 全 -100，只在 JSON 上算 loss
```

#### LoRA 配置

```
Qwen2-0.5B 全参数：495M
LoRA r=8，目标层：q_proj / k_proj / v_proj / o_proj
可训练参数：~1.08M（0.22%）
```

#### 实测结果（全量数据 10748 条，1 epoch，RTX 4060）

| 指标          | 数值               |
|---------------|--------------------|
| Precision     | 0.6351             |
| Recall        | 0.6295             |
| **F1**        | **0.6323**         |
| JSON 解析失败 | **0 条（0%）**     |
| 训练时间      | 3303s（约 55 min） |
| 推理速度      | 1.26s/条（GPU）    |

---

## 4. 实验结果

| 方案                   | Entity F1  | 非法序列 | 评估方式          | 备注                     |
|------------------------|------------|----------|-------------------|--------------------------|
| BERT + Linear          | ~0.79      | 10~30 条 | seqeval（方式 A） | 3 epoch；1 epoch ≈ 72.7% |
| BERT + CRF             | **0.7254** | **0 条** | seqeval（方式 A） | 3 epoch；val 集实测      |
| Qwen API zero-shot     | ~0.55      | -        | span F1（方式 B） | qwen-plus；100 条参考值  |
| Qwen API few-shot      | ~0.63      | -        | span F1（方式 B） | qwen-plus；3 例参考值    |
| Qwen2-0.5B SFT（LoRA） | **0.6323** | **0 条** | span F1（方式 B） | 1 epoch；全量数据；实测  |

**主要结论**：

1. **CRF 的核心价值是合法性保证**：Viterbi 解码将非法序列从 ~20 条降至 0，这是数学保证，不依赖训练轮数。F1 提升约 1
   个点，是额外收益。

2. **SFT ≈ LLM API few-shot，两者均低于 BERT+CRF**：SFT（0.6323）与 Qwen API few-shot（~0.63）处于同一水平，均比
   BERT+CRF（0.7254）低约 9 个点。考虑到 SFT 只训练 1 epoch 且评估方式偏宽，实际差距可能更大。NER
   这类需要精确边界定位的任务，序列标注天然比生成式方法有优势。

3. **SFT 的价值在于灵活性，不在精度**：新增实体类型只需改 system prompt，无需重新标注和训练；且 SFT 的 JSON 解析失败率为
   0%，说明 1 epoch 已能稳定输出格式。

4. **微调小模型（BERT 全量 / Qwen2 0.22% LoRA）均显著优于 LLM API few-shot**：即使是 Qwen2-0.5B 这种参数量 5× BERT
   的模型，few-shot 效果仍低于 BERT 全量微调约 18 个点。

---

## 5. Bad Case 分析

### 5.1 线性头产生的非法序列

```
文本：在深圳华为公司的任正非接受媒体采访
预测：O O B-address I-address B-company I-name O O O O O
                              ↑ B-company 后跟 I-name（类型不一致，非法）

CRF 解码：O O B-address I-address B-company I-company B-name I-name O O O
          ✓ 合法，正确识别两个独立实体
```

### 5.2 难识别的实体类型

- **地址（address）**：边界模糊（"上海市浦东新区"的"新区"是否包含？），F1 通常最低
- **书名/影视（book/movie）**：常被省略书名号，与普通名词混淆
- **政府机构（government）**：与组织机构（organization）界限不清

### 5.3 LLM 典型错误

1. **过度识别**：把描述性词语也标记为实体（hallucination）
2. **类型混淆**：`organization` vs `government` 分不清
3. **边界偏移**：实体文本识别正确，但 text.find () 取到的是错误的出现位置（多次出现时）

---

## 6. 优化方向

### 数据层面

- 数据增强：回译、实体替换（使用同类实体替换）
- 扩充训练集：加入 msra_ner 等数据

### 模型层面

- 换用 roberta-wwm-ext（哈工大全词掩码版）
- span-based NER：直接预测实体起止位置，避免 BIO 错误传播

### 训练策略

- 多任务学习：同时训练多个 NER 数据集
- 类别加权损失：对低频实体（book、game）加大权重
- SFT 增加轮数：当前仅 1 epoch，3 epoch 预计 F1 可继续提升

### 工程部署

- 量化压缩：int8 量化后 BERT 推理速度提升 3-4x
- FastAPI 服务：结合 RAG 系统提取文档中的关键实体

---

## 7. 关键工程决策与踩坑

| 问题                                                        | 根因                                             | 解法                                                 |
|-------------------------------------------------------------|--------------------------------------------------|------------------------------------------------------|
| `hfl/cluener2020` 在 hf-mirror.com 不存在                   | 该数据集未被镜像收录                             | 改用 CLUE 官方 Google Storage URL 直接下载 zip       |
| CLUE cluener2020 test 集标签未公开                          | CLUE 竞赛规则，test 集 label 字段缺失            | 评估统一使用 validation 集（1343 条）                |
| CRF 1 epoch 仍有非法序列                                    | 转移矩阵需充分训练才能学到强约束                 | 训练 3+ epoch 后非法序列趋近 0                       |
| seqeval 对 O 标签的处理                                     | seqeval 要求字符串列表格式 O/B-X/I-X             | 预测 id 先用 `id2label` 转回字符串再传给 seqeval     |
| CRF decode 返回变长列表                                     | `crf.decode()` 返回每条实际 token 数（不含 PAD） | 用 `-100` 过滤 labels 时同步过滤 pred，保证长度对齐  |
| Anaconda numpy 2.x 冲突                                     | scipy/numexpr/bottleneck 用 numpy 1.x 编译       | `pip install --upgrade scipy numexpr bottleneck`     |
| SFT `apply_chat_template(tokenize=True)` 返回 BatchEncoding | transformers 5.x 改变返回类型                    | 改用 `tokenize=False` + 单独 `tokenizer.encode()`    |
| SFT `from_pretrained` 的 `torch_dtype` 废弃                 | transformers 5.x 将参数改名为 `dtype`            | 改用 `dtype=torch.float32`                           |
| NER JSON 输出被截断                                         | 默认 `max_new_tokens` 过小；JSON 比单词长得多    | 评估设 `max_new_tokens=256`，训练设 `max_length=256` |
| 练习数据集 `nlpblogs/CMeEE` 国内无法访问                    | hf-mirror.com 未收录                             | 改用人民日报 NER，从 GitHub raw URL 下载 CoNLL 文件  |

---

## 8. 目录结构

```
bert_ner/
├── src/                         # BERT 序列标注实现
│   ├── download_data.py         # 从 CLUE Google Storage 下载，保存为 JSON
│   ├── explore_data.py          # 实体分布/文本长度可视化
│   ├── dataset.py               # span→BIO + BERT 子词对齐 + DataLoader
│   ├── model.py                 # BertNER（线性）+ BertCRFNER（CRF）
│   ├── train.py                 # 训练（--use_crf 切换模型）
│   ├── evaluate.py              # seqeval F1 + 非法序列统计
│   └── compare_results.py       # 汇总四套方案对比表
│
├── src_llm/                     # LLM 生成式 NER 实现
│   ├── llm_ner.py               # Qwen API zero-shot/few-shot（100条，span F1 评估）
│   ├── train_sft.py             # LoRA 指令微调（chat格式 + loss masking + --full_ft）
│   └── evaluate_sft.py          # 加载 checkpoint（自动识别 LoRA/全量），span F1 四方对比
│
├── data/
│   ├── cluener/                 # train.json / validation.json / test.json
│   └── peoples_daily/           # 课后练习，3类实体（PER/ORG/LOC），CoNLL 格式
│
├── outputs/
│   ├── checkpoints/             # best_linear.pt / best_crf.pt
│   ├── logs/                    # train_*.json / eval_*.json / eval_llm.json / eval_sft.json
│   ├── sft_adapter/             # LoRA adapter（adapter_model.safetensors + adapter_config.json）
│   ├── sft_full_ckpt/           # 全量微调完整模型（--full_ft 时生成）
│   └── figures/                 # 可视化图表
│
├── requirements.txt
├── ARCHITECTURE.md
├── USAGE_GUIDE.md
└── RESUME_GUIDE.md
```
