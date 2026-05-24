# USAGE_GUIDE.md — 代码调用与测试指南

## 一、环境准备

### 1.1 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖版本（已在以下版本验证）：

- torch 2.6.0+cu126
- transformers 5.5.3
- datasets 4.0.0

### 1.2 Tokenizer 确认

项目使用 `bert-base-chinese` tokenizer（本地路径）。

确认本地模型目录存在：

```bash
ls ../../pretrain_models/bert-base-chinese/
# 应看到 vocab.txt、tokenizer_config.json 等文件
```

若不存在，代码会自动从 HuggingFace 下载（需要网络）。

---

## 二、运行步骤

### Step 1：下载数据

```bash
cd src/
python download_data.py
```

**内部流程**：

1. 从 HuggingFace `wikipedia` 数据集加载 `20231101.zh` 快照
2. 取前 50000 篇文章（可用 `--max_articles` 调整）
3. 过滤少于 50 字符的短文章
4. 存为 `data/wiki_zh.jsonl`，每行一个 JSON 对象

**预期输出**：

```
2025-xx-xx xx:xx:xx INFO 加载中文 Wikipedia 数据集（最多 50000 篇）...
2025-xx-xx xx:xx:xx INFO 数据集总篇数：1250000，取前 50000 篇
2025-xx-xx xx:xx:xx INFO 已写入 5000 篇...
...
2025-xx-xx xx:xx:xx INFO 完成！共写入 49xxx 篇 → data/wiki_zh.jsonl
2025-xx-xx xx:xx:xx INFO 文件大小：xxx.x MB
```

**快速验证（仅下载 5000 篇）**：

```bash
python download_data.py --max_articles 5000
```

---

### Step 2：数据预处理

```bash
python prepare_dataset.py
```

**内部流程**：

1. 加载 `bert-base-chinese` tokenizer（vocab_size=21128）
2. 逐篇文章 tokenize，文章间插入 `[SEP]`（token_id=102）
3. 所有 token 拼接成一条长序列
4. 按 `seq_len+1=257` 切块（每块包含 input + target 的右移）
5. 按 95/5 比例分为训练集和验证集
6. 存为 `data/train_seq256.pt` 和 `data/val_seq256.pt`

**预期输出**：

```
使用本地 tokenizer：.../pretrain_models/bert-base-chinese
Tokenizer vocab size: 21128
开始 tokenize 并拼接所有文章...
已处理 5000 篇，当前 token 数：xx,xxx,xxx
...
总 token 数：xxx,xxx,xxx（来自 49xxx 篇文章）
切块数：xxxxxx（seq_len=256）
训练样本：xxxxxx，验证样本：xxxxx
训练集已保存：data/train_seq256.pt（torch.Size([xxxxxx, 257])）
验证集已保存：data/val_seq256.pt（torch.Size([xxxxx, 257])）
```

**快速验证（限制 token 数，约 1 分钟完成预处理）**：

```bash
python prepare_dataset.py --max_tokens 2000000
```

---

### Step 3：训练

```bash
python train.py
```

**内部流程**：

1. 加载 `TokenDataset`，构建 `DataLoader`
2. 构建 MiniGPT（18.8M 参数），打印参数量
3. 计算训练前基线 PPL（随机初始化，约等于 vocab_size=21128）
4. 训练循环：forward → CrossEntropy Loss → backward → clip_grad → optimizer step
5. 每 200 步打印 loss/PPL/lr/grad_norm
6. 每 epoch 结束计算验证集 PPL，保存 checkpoint 和日志

**预期输出**（3 epoch 后）：

```
模型参数量：18.8M
基线 val PPL：21000.0（随机猜测约等于 vocab_size=21128）
Epoch 1/3 Step 200/xxxx | loss=6.5430 PPL=694.2 lr=3.00e-04 grad_norm=0.823
...
Epoch 1 完成 | train PPL=150.3 | val PPL=145.8 | lr=2.85e-04
...
Epoch 3 完成 | train PPL=80.1 | val PPL=95.4 | lr=3.00e-05
最优 val PPL = 95.4
```

**参数调整**：

```bash
# 显存不足时减小 batch_size
python train.py --batch_size 16

# 只跑 1 个 epoch 快速看效果
python train.py --epochs 1

# 调整学习率
python train.py --lr 1e-4
```

---

### Step 4：评估

```bash
python evaluate.py
# 或指定 checkpoint
python evaluate.py --checkpoint outputs/checkpoints/epoch2_ppl100.0.pt
# 同时绘制训练曲线
python evaluate.py --plot
```

**预期输出**：

```
==================================================
验证集评估结果
  总 token 数：x,xxx,xxx
  平均 Cross-Entropy Loss：4.5580
  PPL（困惑度）= exp(4.5580) = 95.4
==================================================
训练曲线已保存：outputs/training_curve.png
```

---

### Step 5：文本生成

```bash
# 对比四种解码策略
python generate.py --compare --prompt "中国的首都是"

# 单策略生成
python generate.py --strategy top_p --prompt "人工智能的发展"
python generate.py --strategy greedy --prompt "北京大学"
python generate.py --strategy temperature --temperature 1.2 --prompt "科技创新"
python generate.py --strategy top_k --top_k 30 --prompt "历史上"
```

**预期输出示例**：

```
======================================================================
Prompt：中国的首都是
======================================================================

【Greedy】
中国的首都是北京，是中华人民共和国的政治、文化中心，位于华北平原...

【Temperature (T=0.8)】
中国的首都是北京市，是全国重要的政治中心，也是国家...

【Top-K (K=50)】
中国的首都是北京，自古以来就是华北地区的重要...

【Top-P (p=0.9)】
中国的首都是北京，北京地处中国北方，是中华人民共和国的首都...
```

---

## 三、作为模块调用

```python
import torch
from pathlib import Path
from transformers import BertTokenizerFast
from src.model import build_model
from src.generate import generate, decode_text

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("outputs/checkpoints/best_model.pt", map_location=device, weights_only=True)
model = build_model(vocab_size=ckpt["vocab_size"], seq_len=ckpt["seq_len"]).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 加载 tokenizer
tokenizer = BertTokenizerFast.from_pretrained("../../pretrain_models/bert-base-chinese")

# 文本生成
prompt = "人工智能"
input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
out_ids = generate(model, input_ids, max_new_tokens=100, strategy="top_p", top_p=0.9)
text = decode_text(tokenizer, out_ids[input_ids.shape[1]:])
print(f"{prompt}{text}")
```

---

## 四、调试与常见问题

**Q: `OMP: Error #15: Initializing libiomp5md.dll`**
A: 所有脚本已在顶部添加 `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")`，若仍报错，手动设置环境变量：

```bash
set KMP_DUPLICATE_LIB_OK=TRUE  # Windows CMD
```

**Q: `OSError: Repo id must use alphanumeric chars`**
A: transformers 5.x 不接受含 `..` 的相对路径。代码已使用 `Path(__file__).parent` 构造绝对路径，如自行修改路径请确保使用绝对路径。

**Q: datasets 报错 `NotImplementedError: Loading a dataset cached`**
A: datasets 4.0 已不支持 dataset script。代码使用 `20231101.zh` parquet 格式快照，确保使用 `datasets>=4.0.0`。

**Q: CUDA out of memory**
A: 减小 `batch_size`（从 32 降到 16 或 8），或减小 `seq_len`（从 256 降到 128）：

```bash
python train.py --batch_size 16 --seq_len 128
```

**Q: PPL 不下降或下降很慢**
A: 检查：(1) 数据量是否太小（`--max_tokens` 是否设置过小）；(2) 学习率是否合适（尝试 `--lr 1e-3` 或 `--lr 1e-4`）；(3) 确认
loss 确实在下降（看 training_log.jsonl）。

**Q: 生成文本出现大量 `[UNK]` 或乱码**
A: bert-base-chinese tokenizer 会将不常见汉字切成 `[UNK]`。这是正常现象，不影响教学演示。

**Q: 训练日志在哪里？**
A: `outputs/training_log.jsonl`，每行一个 epoch 的 JSON 记录，包含 train_loss/train_ppl/val_ppl/lr。
