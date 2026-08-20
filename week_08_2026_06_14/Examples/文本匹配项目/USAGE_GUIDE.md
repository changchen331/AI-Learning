# USAGE_GUIDE.md — 代码调用与测试指南

## 一、环境准备

```bash
pip install torch transformers peft>=0.14.0 scikit-learn matplotlib tqdm openai datasets requests

# API key（LLM API 对比脚本需要）
export DASHSCOPE_API_KEY="sk-xxx"
```

本项目使用以下两个预训练模型：

```
pretrain_models/
├── bert-base-chinese/          ← src/ 使用
└── Qwen2-0.5B-Instruct/        ← src_llm/ 使用（SFT 训练和评估）
```

---

## 二、数据准备

数据已下载完毕，无需重复执行。如需重新下载：

```bash
cd src
python download_data.py
```

**预期输出：**

```
下载 AFQMC（clue/afqmc）...
  train       :  34,334 条  正样本  10,573  负样本  23,761
  validation  :   4,316 条  正样本   1,338  负样本   2,978
  test        :   3,861 条  （CLUE 竞赛格式，test 标签未公开，不用于评估）

下载 LCQMC ...    → data/lcqmc/（学生自主练习）
下载 BQ Corpus ... → data/bq_corpus/（学生自主练习）
```

---

## 三、数据探索

```bash
cd src
python explore_data.py
```

生成 4 张图到 `outputs/figures/`：

| 图表文件                        | 内容                | 教学重点                   |
|---------------------------------|---------------------|----------------------------|
| `label_distribution.png`        | 正/负样本数量       | 类别不均衡（31% vs 69%）   |
| `char_length_distribution.png`  | 字符长度分布        | max_length=32 已覆盖 98.4% |
| `length_diff_distribution.png`  | 正/负样本长度差     | 无 length bias，数据质量好 |
| `token_length_distribution.png` | BERT Token 长度分布 | Token ≈ 字符（中文特性）   |

---

## 四、训练 BiEncoder（表示型，重点）

### 4.1 CosineEmbeddingLoss 训练

```bash
cd src
python train_biencoder.py --loss cosine
```

默认参数：`--pool mean --num_hidden_layers 4 --epochs 3 --batch_size 32 --lr 2e-5 --margin 0.3`

**内部流程：**

1. 加载 AFQMC train/val（PairDataset，sentence1 / sentence2 / label）
2. 每个 step：encode (s1) → emb_a，encode (s2) → emb_b，label 0→-1 / 1→+1
3. `F.cosine_embedding_loss(emb_a, emb_b, cos_target, margin=0.3)`
4. 每个 epoch 末：val 集计算余弦相似度 → 枚举 101 个阈值 → 取 F1 最高
5. 保存 val_f1 最优的 checkpoint

**预期输出（每 epoch）：**

```
Epoch 1/3 | train_loss=0.1234 | val_acc=0.7500 val_f1=0.7234 threshold=0.73 | 35s
  ✓ 新最优模型已保存 → .../outputs/checkpoints/biencoder_cosine_best.pt
```

### 4.2 TripletLoss 训练

```bash
python train_biencoder.py --loss triplet --margin 0.3
```

**内部流程（与 cosine 的差异）：**

1. 使用 TripletDataset：从 10,573 个正例对构建三元组（anchor, positive, negative）
2. 每个 step：`F.triplet_margin_loss(emb_a, emb_p, emb_n, margin=0.3)`
3. 评估仍用 PairDataset（余弦相似度 + 阈值搜索）

**验证两种 Loss 的差异：** 对比 `outputs/logs/biencoder_cosine_log.json` 与 `biencoder_triplet_log.json` 中的 `val_f1`
曲线。

### 4.3 参数建议（课堂演示 vs 学生练习）

| 场景         | 推荐参数                                            |
|--------------|-----------------------------------------------------|
| 课堂快速演示 | `--num_hidden_layers 4 --epochs 3 --batch_size 32`  |
| 学生完整训练 | `--num_hidden_layers 12 --epochs 5 --batch_size 16` |
| 池化策略对比 | `--pool cls` vs `--pool mean` vs `--pool max`       |

---

## 五、训练 CrossEncoder（交互型，对比）

```bash
cd src
python train_crossencoder.py
```

默认参数：`--num_hidden_layers 4 --epochs 3 --batch_size 32`

**与 BiEncoder 的关键差异：**

- 输入：`tokenizer(sentence1, sentence2)` 生成 `[CLS] s1 [SEP] s2 [SEP]`
- 评估：直接 `argmax(logits)`，无需阈值搜索
- 训练更慢（max_length=128 vs BiEncoder 的 64）

**预期输出：**

```
Epoch 1/3 | train_loss=0.4812 train_acc=0.7234 | val_acc=0.7681 val_f1=0.7412 | 52s
```

---

## 六、评估（加载 checkpoint）

```bash
cd src
# BiEncoder
python evaluate.py --model_type biencoder \
  --ckpt ../outputs/checkpoints/biencoder_cosine_best.pt

# CrossEncoder
python evaluate.py --model_type crossencoder \
  --ckpt ../outputs/checkpoints/crossencoder_best.pt
```

**BiEncoder 额外输出：** 相似度分布图 `outputs/figures/biencoder_validation_sim_dist.png`

---

## 七、方法对比（三种训练方式）

```bash
cd src
# 确保三种方法都已训练完（各 1 epoch）后运行
python compare_methods.py
```

**前提：** `outputs/checkpoints/` 下需要有以下三个文件：

- `biencoder_cosine_best.pt`（`python train_biencoder.py --loss cosine`）
- `biencoder_triplet_best.pt`（`python train_biencoder.py --loss triplet`）
- `crossencoder_best.pt`（`python train_crossencoder.py`）

**输出示例（4 层 × 1 epoch）：**

```
方法                              Accuracy  F1(weighted)    额外信息
BiEncoder (CosineEmbeddingLoss)    0.6643        0.6505  threshold=0.55
BiEncoder (TripletLoss)            0.6569        0.6286  threshold=0.84
CrossEncoder (CrossEntropyLoss)    0.6921        0.5703          argmax
```

**几个值得注意的规律：**

- CrossEncoder Accuracy 最高但 F1 最低：1 epoch 训练不足时倾向于预测多数类（负类）， accuracy
  因此虚高——这本身就是一个教学点（accuracy ≠ F1）
- CosineEmbeddingLoss 优于 TripletLoss：AFQMC 正样本只有 10K 条，TripletLoss 三元组数量受限；数据量更大时（如 LCQMC）Triplet
  的优势会更明显
- 生成图表：`method_comparison_bar.png`（柱状对比）+ `biencoder_sim_distributions.png`（分布对比）

---

## 八、Bad Case 分析与优化方向

```bash
cd src
# 分析 BiEncoder（CosineEmbeddingLoss）的错误案例
python analyze_badcases.py

# 分析 CrossEncoder
python analyze_badcases.py --model_type crossencoder \
  --ckpt ../outputs/checkpoints/crossencoder_best.pt

# 展示更多案例
python analyze_badcases.py --n_cases 10
```

**输出内容：**

1. **FP/FN 汇总**：按错误类型和置信度分级（高置信度错误 vs 临界错误）
2. **语言特征分析**：长度差、字符 Jaccard 相似度（揭示错误根因）
3. **典型案例展示**：高置信度错误最具教学价值
4. **优化方向建议**：数据、模型、训练策略、部署四个层面
5. **Score 分布图**：`biencoder_badcase_dist.png`（正确 vs 错误的分数分布）

**关键发现（实测结果）：**

```
FP (假阳性) 567条：字符 Jaccard 均值=0.506  → 词汇高度重叠但语义不同
FN (假阴性) 880条：字符 Jaccard 均值=0.388  → 换了表达方式，词汇重叠低
```

这直接指向两个优化方向：FP → 增大 margin + 难负样本挖掘；FN → SimCSE 对比学习 / 更多层数

---

## 九、LLM zero-shot 对比（API 方式）

```bash
cd src_llm
export DASHSCOPE_API_KEY="sk-xxx"
python llm_compare.py --num_samples 100 --model qwen-plus
```

**说明：**

- 默认只评估 100 条（约消耗 ¥0.1），足够展示效果差异
- 输出包含 Accuracy / F1（正例），以及与 BERT 的对比表
- 结果自动保存到 `outputs/logs/llm_compare_results.json`，供 evaluate_sft.py 读取

---

## 十、LLM SFT 指令微调（LoRA / 全量微调）

```bash
cd src_llm

# ── LoRA 微调（默认，推荐演示）──────────────────────────────────────────────
python train_sft.py                        # 5000 条，3 epoch（快速演示）
python train_sft.py --num_train -1         # 全部 34K 条
python train_sft.py --epochs 1             # 1 epoch 快速验证

# ── 全量微调（需显存 ≥ 16GB）────────────────────────────────────────────────
python train_sft.py --full_ft --lr 2e-5
```

**完整参数说明**：

```bash
python train_sft.py \
  --num_train 5000    \  # 训练样本数，-1 用全部 34K 条
  --epochs 3          \  # 训练轮数
  --batch_size 4      \  # 每步 batch 大小
  --grad_accum 4      \  # 梯度累积，等效 batch = 16
  --max_length 128    \  # 句对 + 模板 < 128，无需 256
  --lora_r 8          \  # LoRA rank（仅 LoRA 模式有效）
  --full_ft              # 切换为全量微调（默认 LoRA）
```

**两种模式对比**：

| 维度            | LoRA（默认）                  | 全量微调（`--full_ft`）           |
|-----------------|-------------------------------|-----------------------------------|
| 可训练参数      | ~1.08M（0.22%）               | 495M（100%）                      |
| 默认学习率      | 2e-4（自动）                  | 2e-5（需手动指定）                |
| checkpoint 目录 | `outputs/sft_adapter/`        | `outputs/sft_full_ckpt/`          |
| 日志文件        | `outputs/logs/train_sft.json` | `outputs/logs/train_full_ft.json` |

> **类别平衡说明**：AFQMC 正负比 31:69。`train_sft.py` 默认开启 **正负平衡采样**（各取 `num_train//2` 条），避免模型退化为全预测负例（F1=0
> 的教学反例在第一版实测中出现过，与 CrossEncoder 1-epoch 是同一问题）。

---

## 十一、SFT 模型评估

```bash
cd src_llm

python evaluate_sft.py                                     # 评估 LoRA（默认，200 条）
python evaluate_sft.py --ckpt_dir ../outputs/sft_full_ckpt  # 评估全量微调
python evaluate_sft.py --demo                              # 5 条快速演示
python evaluate_sft.py --num_samples 500                   # 更多样本
```

`evaluate_sft.py` 自动识别 checkpoint 类型，并读取 `llm_compare_results.json` 和 BERT 训练日志做多方对比。

**实测输出**（LoRA，平衡采样 5000 条，1 epoch，200 条评估，seed=42）：

```
样本数      : 200（有效: 200，parse_fail: 0）
Accuracy    : 0.6400
F1(weighted): 0.6535
F1(正例)    : 0.5556
均值耗时    : 0.10s/条（GPU）

多方对比（AFQMC validation 集，所有方案均使用 Accuracy + F1，直接可比）
  BiEncoder + CosineEmbeddingLoss    F1(pos) = 0.6505
  BiEncoder + TripletLoss            F1(pos) = 0.6286
  CrossEncoder + CrossEntropyLoss    F1(pos) = 0.5703
  Qwen API zero-shot                 F1(pos) = ?（运行 llm_compare.py 后获取）
  Qwen2-0.5B SFT（LoRA，5K 平衡）   F1(pos) = 0.5556
```

> SFT 与 BERT 方法使用完全相同的 Accuracy + F1 指标，无评估标准差异，数字可直接比较。

---

## 十、作为模块调用

```python
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as f
from pathlib import Path
from transformers import BertTokenizer
from model import build_biencoder

# 初始化
BERT_PATH = "E:/badou/项目材料准备/pretrain_models/bert-base-chinese"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

# 加载训练好的模型
ckpt = torch.load("outputs/checkpoints/biencoder_cosine_best.pt",
                  map_location=device, weights_only=False)
model = build_biencoder(BERT_PATH, pool="mean", num_hidden_layers=4).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()


# 单次推理
def encode(text):
    enc = tokenizer(text, max_length=64, truncation=True,
                    padding="max_length", return_tensors="pt")
    return model.encode(
        enc["input_ids"].to(device),
        enc["attention_mask"].to(device),
        enc["token_type_ids"].to(device),
    )


s1 = "花呗怎么还款"
s2 = "如何偿还花呗账单"
emb1 = encode(s1)
emb2 = encode(s2)
sim = f.cosine_similarity(emb1, emb2).item()
threshold = ckpt["threshold"]  # 训练时在 val 集搜出的最优阈值
print(f"相似度: {sim:.4f}  阈值: {threshold:.2f}  预测: {'相似' if sim >= threshold else '不相似'}")
```

---

## 十一、调试与常见问题

**Q: `OMP: Error #15: Initializing libiomp5md.dll`**
> 已在所有脚本顶部加 `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` 修复

**Q: `OSError: Repo id must use alphanumeric chars`**
> transformers 将相对路径当成 HuggingFace repo ID 验证。已用 `Path(__file__).parent.parent` 构造绝对路径解决

**Q: BiEncoder 评估时 val_f1 很低（< 0.6）**
> 可能原因： (1) 学习率偏高（试 `--lr 1e-5`）； (2) epoch 不够（试 `--epochs 5`）； (3) margin 过大（试 `--margin 0.1`）

**Q: AFQMC test 集评估结果异常（全预测为 0）**
> 正常现象——test 集标签在 CLUE 竞赛中未公开，label=-1。评估请用 `--split validation`

**Q: TripletLoss 训练 loss 不下降**
> AFQMC 三元组约 1 万条（正例数量限制），数据量较小。可调小 `--margin 0.1` 或改用全量 12 层

**Q: LLM 对比脚本报 `DASHSCOPE_API_KEY` 未设置**
> 运行 `export DASHSCOPE_API_KEY="sk-xxx"` 后再执行

**Q: train_sft.py 报 `ModuleNotFoundError: No module named 'peft'`**
> LoRA 微调需要 peft 库：`pip install peft>=0.14.0`。全量微调（`--full_ft`）不需要。

**Q: evaluate_sft.py 报 checkpoint 目录不存在**
> 需先运行 `train_sft.py` 完成训练。LoRA 保存到 `outputs/sft_adapter/`，全量保存到 `outputs/sft_full_ckpt/`。

**Q: SFT 的 parse_fail 率较高**
> 说明训练不足。文本匹配的 TARGET 只有 3~5 token，一般训练 1 epoch 就能稳定输出。检查 SYSTEM_PROMPT 是否与 LABEL_MAP
> 一致（均使用【相似】/【不相似】）。
