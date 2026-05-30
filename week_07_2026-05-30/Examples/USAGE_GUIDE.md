# USAGE_GUIDE.md — 代码调用与测试指南

## 1. 环境准备

### 依赖安装

```bash
pip install -r requirements.txt
# 核心依赖：
# pytorch-crf>=0.7.2  —— CRF 层
# seqeval>=1.2.2       —— entity-level F1 评估
# peft>=0.14.0         —— LoRA 微调（train_sft.py 依赖，全量微调模式不需要）
```

### API Key 配置

```bash
# Windows CMD
set DASHSCOPE_API_KEY=sk-xxx

# Windows PowerShell
$env:DASHSCOPE_API_KEY = "sk-xxx"

# 或写入 .env（不提交 git）
DASHSCOPE_API_KEY=sk-xxx
```

### 本地模型路径

本项目使用以下两个预训练模型（均在 `../pretrain_models/` 目录下）：

```
pretrain_models/
├── bert-base-chinese/          ← src/train.py 和 evaluate.py 使用
│   ├── config.json
│   ├── vocab.txt
│   └── pytorch_model.bin（或 model.safetensors）
└── Qwen2-0.5B-Instruct/        ← src_llm/train_sft.py 和 evaluate_sft.py 使用
    ├── config.json
    ├── tokenizer.json
    └── model.safetensors
```

如需指定其他路径：BERT 脚本支持 `--bert_path`，SFT 脚本支持 `--model_path`。

---

## 2. 各步骤流程

### Step 1：下载数据

```bash
cd src/
python download_data.py                       # 下载 cluener2020 + 人民日报 NER
python download_data.py --skip_peoples_daily  # 只下载主数据集
```

**内部流程**：

1. 从 `https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip` 下载约 1MB 的 zip
2. 解析 `train.json` / `dev.json` / `test.json`（每行一个 JSON 对象）
3. 保存为 `data/cluener/{train,validation,test}.json`（indent=2，便于人工检查）
4. 从 GitHub 下载人民日报 NER 三个 split（CoNLL 格式，空行分隔句子），解析后保存到 `data/peoples_daily/`

**预期输出**：

```
[train]      10748 条 → data/cluener/train.json
[validation]  1343 条 → data/cluener/validation.json
[test]        1345 条 → data/cluener/test.json
实体类型（共10类）：['address', 'book', 'company', ...]

[train]      20864 条 → data/peoples_daily/train.json
[validation]  2318 条 → data/peoples_daily/validation.json
[test]        4636 条 → data/peoples_daily/test.json
人民日报 NER 标签体系（共7个）：['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
```

---

### Step 2：探索数据

```bash
python explore_data.py
```

**输出文件**：

- `outputs/figures/entity_distribution.png` — 各类实体频次直方图
- `outputs/figures/text_length_distribution.png` — 文本长度分布（含P95线）
- `outputs/figures/entity_length_distribution.png` — 实体字符数分布

**关键观察**：

- 人名（name）和公司（company）频次最高，书名/游戏频次较低
- P95 文本长度约 60 字，max_length=128 足够覆盖

---

### Step 3：训练 BERT + Linear（基线）

```bash
python train.py                             # 默认 3 epochs，batch=32
python train.py --epochs 5 --lr 3e-5       # 更多轮次
python train.py --batch_size 16            # 显存不足时
```

**关键参数**：

| 参数               | 默认值  | 说明                   |
|------------------|------|----------------------|
| `--epochs`       | 3    | 建议 3~5 轮             |
| `--batch_size`   | 32   | 24GB GPU可用32，8GB建议16 |
| `--max_length`   | 128  | 文本截断长度               |
| `--lr`           | 2e-5 | BERT层学习率             |
| `--head_lr_mult` | 5.0  | 分类头学习率倍数             |
| `--warmup_ratio` | 0.1  | 预热步比例                |

**预期输出**（3 epochs）：

```
Epoch 1/3 | train_loss=0.42 | val_loss=0.20 | val_entity_f1=0.7273 | time=142s
Epoch 2/3 | train_loss=0.20 | val_loss=0.16 | val_entity_f1=0.7720 | time=140s
Epoch 3/3 | train_loss=0.13 | val_loss=0.15 | val_entity_f1=0.7910 | time=141s
★ 最优 val_entity_f1，checkpoint 已保存
```

**输出文件**：

- `outputs/checkpoints/best_linear.pt`
- `outputs/logs/train_linear.json`

---

### Step 4：训练 BERT + CRF

```bash
python train.py --use_crf                   # 默认 3 epochs
python train.py --use_crf --batch_size 16  # 显存不足时
```

CRF 版每个 epoch 比 Linear 版慢约 20-30%（前向-后向算法开销）。

**预期输出**（3 epochs）：

```
Epoch 1/3 | train_loss=0.13 | val_loss=0.09 | val_entity_f1=0.7350 | time=195s
Epoch 2/3 | train_loss=0.07 | val_loss=0.08 | val_entity_f1=0.7780 | time=190s
Epoch 3/3 | train_loss=0.05 | val_loss=0.08 | val_entity_f1=0.8020 | time=192s
```

**输出文件**：

- `outputs/checkpoints/best_crf.pt`
- `outputs/logs/train_crf.json`

---

### Step 5：评估两个模型

```bash
# 在验证集上评估（CLUE cluener2020 test 集标签未公开，使用 validation）
python evaluate.py                  # 评估 BERT+Linear（默认 validation）
python evaluate.py --use_crf        # 评估 BERT+CRF（默认 validation）

# 如需测试集（无标签，F1=0，仅用于查看非法序列分布）
python evaluate.py --split test
```

**输出示例（BERT+Linear）**：

```
Entity-level Precision: 0.8121
Entity-level Recall:    0.7730
Entity-level F1:        0.7921

【逐类型 F1】
          precision  recall  f1-score  support
 address     0.72      0.65     0.68      380
    book     0.83      0.79     0.81       95
 company     0.82      0.78     0.80      400
...

【非法 BIO 序列统计】
  总序列数：1345
  非法开头（I-X 开头）：3 条
  非法转移（B-X/I-X → I-Y, X≠Y）：18 条
  合计非法序列：21 条
  → 线性头约 1.6% 的序列含非法转移，CRF 可完全消除
```

**输出文件**：

- `outputs/logs/eval_linear_validation.json`
- `outputs/logs/eval_crf_validation.json`

---

### Step 6：LLM NER 对比（API 方式）

```bash
cd ../src_llm/

python llm_ner.py                     # 默认 100 条，qwen-plus
python llm_ner.py --n_samples 50      # 减少样本数，快速体验
python llm_ner.py --model qwen-max    # 使用更强模型
```

**费用估算**：qwen-plus 100 条 × (约 200 input + 100 output tokens) × 2（zero+few）≈ **¥0.3~0.5**

**评估方式**：span F1 + text.find() 近似定位（与 evaluate_sft.py 完全一致，数字可直接对比）

**预期输出**：

```
已处理 10/100 条 ...
LLM NER 对比结果（模型：qwen-plus，样本：100 条）
    方案             Precision     Recall        F1
------------------------------------------------------
    Zero-shot         0.5800       0.5200      0.5500
    Few-shot (3例)    0.6500       0.5800      0.6100
```

**输出文件**：`outputs/logs/eval_llm.json`

---

### Step 7：LLM SFT 指令微调（LoRA / 全量微调）

```bash
cd ../src_llm/

# ── LoRA 微调（默认，推荐演示）──────────────────────────────────────────────
python train_sft.py                            # 全部 10748 条，RTX 4060 约 55 min
python train_sft.py --num_train 2000 --epochs 1  # 快速演示

# ── 全量微调（需显存 ≥ 16GB）────────────────────────────────────────────────
python train_sft.py --full_ft --lr 2e-5
```

**完整参数说明**：

```bash
python train_sft.py \
  --num_train -1      \  # 训练样本数，-1 用全部 10748 条（默认）
  --epochs 3          \  # 训练轮数（当前实测 1 epoch）
  --batch_size 4      \  # 每步 batch 大小
  --grad_accum 4      \  # 梯度累积，等效 batch = 16
  --max_length 256    \  # NER JSON 输出比分类长，必须用 256（分类用 128 即可）
  --lora_r 8          \  # LoRA rank（仅 LoRA 模式有效）
  --full_ft              # 切换为全量微调（默认 LoRA）
```

**两种模式对比**：

| 维度            | LoRA（默认）                      | 全量微调（`--full_ft`）                 |
|---------------|-------------------------------|-----------------------------------|
| 可训练参数         | ~1.08M（0.22%）                 | 495M（100%）                        |
| 默认学习率         | 2e-4（自动）                      | 2e-5（需手动指定）                       |
| 显存需求          | ~3GB                          | ~8~10GB                           |
| checkpoint 目录 | `outputs/sft_adapter/`        | `outputs/sft_full_ckpt/`          |
| 日志文件          | `outputs/logs/train_sft.json` | `outputs/logs/train_full_ft.json` |

**实测训练日志**（LoRA，全量数据，1 epoch，RTX 4060）：

```
trainable params: 1,081,344 || all params: 495,114,112 || trainable%: 0.2184
总训练步数: 671（batch=4, grad_accum=4, epochs=1, lr=0.0002）

Epoch 1/1 | train_loss=0.1111  val_loss=0.0941 | 3303s
  ✓ 最优 LoRA adapter 已保存 (val_loss=0.0941)
```

---

### Step 8：SFT 模型评估

```bash
cd ../src_llm/

python evaluate_sft.py                                    # 评估 LoRA（默认）
python evaluate_sft.py --ckpt_dir ../outputs/sft_full_ckpt  # 评估全量微调
python evaluate_sft.py --demo                             # 5 条快速演示
```

`evaluate_sft.py` 自动识别 checkpoint 类型（目录含 `adapter_config.json` → LoRA，否则 → 全量），并与 `eval_llm.json` 中的
API 结果自动对比。

**实测输出**（LoRA，全量数据 1 epoch，100 条，seed=42）：

```
模型加载完成（LoRA adapter 已合并）

[  1/100] ~  gold:先知摩西的纪念馆(scene),加利利湖(scene)  |  在day4去完马达巴之后...
[  2/100] ~  gold:dota(game)                               |  玩dota，你会发现...
...
LLM SFT NER 评估结果
  样本数      : 100
  Precision   : 0.6351
  Recall      : 0.6295
  F1          : 0.6323
  JSON 解析失败: 0 条 (0.0%)
  均值耗时    : 1.26s/条（GPU）

四方对比（span F1，seqeval 与 span F1 差异 < 0.01，基本可比）
  BERT + CRF（3 epoch）        F1 = 0.7254  [seqeval，BIO 精确边界]
  Qwen API zero-shot（100 条） F1 = ~0.55   [span F1 + text.find()]
  Qwen API few-shot（100 条）  F1 = ~0.63   [span F1 + text.find()]
  Qwen2-0.5B SFT（LoRA，1ep） F1 = 0.6323  [span F1 + text.find()]
```

> **关键观察**：SFT（0.6323）与 Qwen API few-shot（~0.63）处于同一水平，两者均比 BERT+CRF（0.7254）低约 9 个点。
> SFT 的主要优势是：训练后格式输出稳定（parse_fail=0%），且可以在本地离线使用，不依赖 API。

---

### Step 9：汇总对比

```bash
cd ../src
python compare_results.py
```

**预期输出**：

```
BERT NER 项目 — 四方案汇总对比
方案                      Precision   Recall      F1      非法序列   评估方式
BERT + Linear              ~0.81      ~0.77     ~0.79       ~20     seqeval
BERT + CRF                  0.82       0.79      0.7254       0     seqeval
Qwen API zero-shot         ~0.58      ~0.52     ~0.55       N/A    span F1
Qwen API few-shot          ~0.65      ~0.58     ~0.63       N/A    span F1
Qwen2-0.5B SFT (LoRA)      0.6351     0.6295    0.6323        0    span F1
```

---

## 3. 作为模块调用

```python
from pathlib import Path
import torch
from transformers import BertTokenizer

# 路径设置（假设工作目录是项目根目录）
import sys

sys.path.insert(0, str(Path("src")))

from dataset import build_label_schema
from model import build_model

# 初始化
labels, label2id, id2label = build_label_schema()
tokenizer = BertTokenizer.from_pretrained("../pretrain_models/bert-base-chinese")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 BERT+CRF checkpoint
model = build_model(use_crf=True, bert_path="../pretrain_models/bert-base-chinese",
                    num_labels=len(labels))
ckpt = torch.load("outputs/checkpoints/best_crf.pt", weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.to(device).eval()

# 单条推理
text = "华为技术有限公司总裁任正非在深圳接受媒体采访"
chars = list(text)
enc = tokenizer(chars, is_split_into_words=True, max_length=128,
                truncation=True, padding="max_length", return_tensors="pt")
input_ids = enc["input_ids"].to(device)
attention_mask = enc["attention_mask"].to(device)
token_type_ids = enc["token_type_ids"].to(device)

with torch.no_grad():
    pred_ids = model.decode(input_ids, attention_mask, token_type_ids)[0]

# 对齐 id2label（跳过 [CLS] [SEP]）
word_ids = enc.word_ids(0)
results = []
prev_entity = None
for j, wid in enumerate(word_ids):
    if wid is None:
        continue
    if j < len(pred_ids):
        tag = id2label[pred_ids[j]]
        if tag.startswith("B-"):
            prev_entity = {"type": tag[2:], "start": wid, "end": wid, "text": chars[wid]}
            results.append(prev_entity)
        elif tag.startswith("I-") and prev_entity and prev_entity["type"] == tag[2:]:
            prev_entity["end"] = wid
            prev_entity["text"] = text[prev_entity["start"]:prev_entity["end"] + 1]
        else:
            prev_entity = None

print(results)
# 输出：[{"type": "company", "start": 0, "end": 7, "text": "华为技术有限公司"},
#        {"type": "position", "start": 8, "end": 9, "text": "总裁"},
#        {"type": "name", "start": 10, "end": 12, "text": "任正非"},
#        {"type": "address", "start": 14, "end": 15, "text": "深圳"}]
```

---

## 4. 调试与常见问题

**Q: `DatasetNotFoundError: Dataset 'hfl/cluener2020' doesn't exist`**
A: 本项目已改为从 CLUE 官方 Google Storage 下载，`download_data.py` 不走 HuggingFace。

**Q: 为什么练习数据集从 CMeEE 换成了人民日报 NER？**
A: `nlpblogs/CMeEE` 未被 hf-mirror.com 收录，国内无法通过 `load_dataset` 下载。人民日报 NER 格式更简洁（CoNLL，3类实体），从
GitHub raw URL 直接下载，无需任何账号，适合作为练习补充。数据位于 `data/peoples_daily/`，标签体系：
`O / B-PER / I-PER / B-ORG / I-ORG / B-LOC / I-LOC`。

**Q: `ImportError: numpy.core.multiarray failed to import`**
A: Anaconda 的 scipy/numexpr/bottleneck 与 numpy 2.x 冲突，运行：

```bash
pip install --upgrade scipy numexpr bottleneck pyarrow
```

**Q: CRF 训练比 Linear 慢很多**
A: 正常现象。CRF 前向算法需要 O(L × K²) 计算（L=序列长，K=标签数）。
对于 max_length=128、21个标签，每步约慢 20-30%。

**Q: seqeval 报错 `ValueError: prefixes are wrong`**
A: 确认标签字符串格式为 `O` / `B-type` / `I-type`，不能有 `B_type`（下划线分隔）。

**Q: CRF 的 `emissions` 形状错误**
A: `torchcrf` 要求 `batch_first=True` 时形状为 `(batch, seq_len, num_labels)`，
mask 必须是 `BoolTensor`（`attention_mask.bool()`）。

**Q: 显存不足（OOM）**
A: 调低 `--batch_size`（16 → 8），或者降低 `--max_length`（128 → 64）。
cluener2020 的 P95 文本长度约 60 字，max_length=64 损失信息极少。

**Q: train_sft.py 报 `ModuleNotFoundError: No module named 'peft'`**
A: LoRA 微调需要 peft 库，运行：

```bash
pip install peft>=0.14.0
```

全量微调模式（`--full_ft`）不需要 peft。

**Q: evaluate_sft.py 报 "checkpoint 目录不存在"**
A: 需要先运行 `train_sft.py` 完成训练。LoRA 默认保存到 `outputs/sft_adapter/`，全量微调保存到 `outputs/sft_full_ckpt/`。

**Q: SFT 的 F1（0.63）低于 BERT+CRF（0.73），差在哪里？**
A: 两者的评估标准已基本统一（均为 span F1 + text.find() 定位，seqeval 与之差 < 0.01），差距是真实的。根本原因是 NER 任务的特性：

1. 序列标注天然适合精确边界定位——BIO 标签逐 token 输出，边界信息直接编码在标签里
2. CRF 的 Viterbi 解码还保证零非法序列，生成式方法没有这个保证
3. 当前 SFT 只训了 1 epoch，更多轮次会继续提升

**Q: SFT 和 LLM API few-shot 的 F1 差不多，SFT 有什么意义？**
A: F1 数字接近（0.63 vs 0.63），但两者有本质区别：

1. **离线 vs 在线**：SFT 模型本地运行，无 API 调用成本和延迟，数据不出本地
2. **格式稳定性**：SFT 后 parse_fail=0%（已学会输出格式），API 模式在复杂句子上偶尔输出乱格式 JSON
3. **可继续优化**：SFT 可以通过更多 epoch、数据增强继续提升，few-shot 的上限基本固定

**Q: JSON 解析失败率较高（parse_fail > 10%）**
A: 生成式 NER 的固有问题。Qwen2-0.5B 模型较小，训练不足时输出格式不稳定。改进方法：

- 增加训练轮数（`--epochs 5`）
- 在 system prompt 里增加格式示例（few-shot SFT）
- 这个不稳定性本身是教学点：对比 BERT+CRF 的确定性输出
