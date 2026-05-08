"""
train_position_cls_compare.py
语序信息对比实验 —— "你"在第几位就属于第几类

任务：随机生成含"你"的 5 字句子，"你"在第 pos 位（1-indexed 类别 1~5）
      每个类别对"词袋模型"而言完全无法区分（相同字符集，只是顺序不同）

模型A（RNN + MaxPool）：Embedding → RNN → MaxPool → Linear
  RNN 逐字处理，每步隐藏状态携带"已读到哪"的上下文，池化后仍保留语序信息

模型B（BoW  MaxPool）：Embedding → MaxPool → Linear   （不使用 RNN）
  直接对字符 Embedding 做 MaxPool ≈ 词袋，输出与字符顺序无关，
  理论上完全无法判断"你"的位置，准确率应接近随机基准（20%）

运行结果对比展示语序信息对 NLP 任务的重要性。

依赖：torch >= 2.0   (pip install torch)
"""

import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 10000  # 总样本数；每类 2000 条
SEQ_LEN = 5  # 句子固定 5 字
EMBED_DIM = 64
HIDDEN_DIM = 64
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 20
TRAIN_RATIO = 0.8
NUM_CLASSES = 5  # "你"的位置：第 1~5 位对应类别 0~4

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 字符池（不含"你"）──────────────────────────────────────────────
_RAW = (
    '的一是了我不人在他有这个上们来到时大地为子中说生国年着'
    '就那和要她出也得里后自以会家可下而过天去能对小多然于心'
    '学么之都好看起发当没成只如事把还用第样道想作种开美总从'
    '无情己面最女但现前些所同日手又行意别信走定回爱进此'
)
CHAR_POOL = list(set(ch for ch in _RAW if ch != '你'))


# ─── 2. 数据生成 ───────────────────────────────────────────────────────
def make_sample(pos: int):
    """生成一条 5 字样本，"你"固定在 pos 位（0-indexed），标签 = pos"""
    chars = random.choices(CHAR_POOL, k=SEQ_LEN - 1)
    chars.insert(pos, '你')
    return ''.join(chars), pos


def build_dataset(n: int = N_SAMPLES):
    per_class = n // NUM_CLASSES
    data = []
    for cls in range(NUM_CLASSES):
        for _ in range(per_class):
            data.append(make_sample(cls))
    random.shuffle(data)
    return data


# ─── 3. 词表与编码 ─────────────────────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab):
    ids = [vocab.get(ch, 1) for ch in sent[:SEQ_LEN]]
    ids += [0] * (SEQ_LEN - len(ids))
    return ids


# ─── 4. Dataset ────────────────────────────────────────────────────────
class PositionDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )


# ─── 5. 模型定义 ───────────────────────────────────────────────────────
class RNNModel(nn.Module):
    """
    模型A：Embedding → RNN → MaxPool（沿时序）→ Linear

    RNN 按位置顺序逐字处理，第 t 步的隐藏状态 h_t 编码了
    "前 t 个字"的上下文；MaxPool 汇聚全局最显著特征，
    但特征本身已携带位置信息。
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.rnn = nn.RNN(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, x):
        # x      : (B, SEQ_LEN)
        emb = self.embedding(x)  # (B, SEQ_LEN, EMBED_DIM)
        rnn_out, _ = self.rnn(emb)  # (B, SEQ_LEN, HIDDEN_DIM)
        pooled = rnn_out.max(dim=1)[0]  # (B, HIDDEN_DIM)  时序 max-pool
        return self.fc(pooled)  # (B, NUM_CLASSES)


class BagModel(nn.Module):
    """
    模型B：Embedding → MaxPool（沿时序）→ Linear   【不使用 RNN】

    直接对字符 Embedding 做 MaxPool，结果与字符顺序无关，
    相当于"词袋(Bag-of-Characters)"表示，无法区分"你"所在位置。
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.fc = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x):
        # x      : (B, SEQ_LEN)
        emb = self.embedding(x)  # (B, SEQ_LEN, EMBED_DIM)
        pooled = emb.max(dim=1)[0]  # (B, EMBED_DIM)  顺序无关！
        return self.fc(pooled)  # (B, NUM_CLASSES)


# ─── 6. 训练 & 评估工具 ────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def evaluate_per_class(model, loader):
    model.eval()
    correct = [0] * NUM_CLASSES
    total = [0] * NUM_CLASSES
    with torch.no_grad():
        for X, y in loader:
            pred = model(X).argmax(dim=1)
            for cls in range(NUM_CLASSES):
                mask = (y == cls)
                correct[cls] += (pred[mask] == cls).sum().item()
                total[cls] += mask.sum().item()
    return [correct[c] / total[c] if total[c] else 0.0 for c in range(NUM_CLASSES)]


def train_model(model, train_loader, val_loader, name):
    print(f"\n{'=' * 56}")
    print(f"  训练：{name}")
    print(f"  参数量：{sum(p.numel() for p in model.parameters()):,}")
    print(f"{'=' * 56}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 4 == 0 or epoch == 1:
            avg_loss = total_loss / len(train_loader)
            val_acc = evaluate(model, val_loader)
            print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    final_acc = evaluate(model, val_loader)
    print(f"  最终验证准确率：{final_acc:.4f}")
    return final_acc, model


# ─── 7. 主流程 ─────────────────────────────────────────────────────────
def main():
    print("─── 数据集准备 ────────────────────────────────────────────────")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  总样本：{len(data)}，每类 {len(data) // NUM_CLASSES} 条")
    print(f"  词表大小：{len(vocab)}")
    print(f'  任务：预测「你」在 5 字句子中的位置（第 1~5 位，{NUM_CLASSES} 分类）')
    print(f"  随机猜测基准准确率：{1 / NUM_CLASSES:.0%}")

    split = int(len(data) * TRAIN_RATIO)
    train_loader = DataLoader(
        PositionDataset(data[:split], vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        PositionDataset(data[split:], vocab), batch_size=BATCH_SIZE)

    rnn_acc, rnn_model = train_model(
        RNNModel(len(vocab)), train_loader, val_loader,
        "模型A — RNN + MaxPool（保留语序）"
    )
    bow_acc, bow_model = train_model(
        BagModel(len(vocab)), train_loader, val_loader,
        "模型B — 直接 Embedding MaxPool（无 RNN，丢失语序）"
    )

    # ─── 总体对比 ──────────────────────────────────────────────────────
    print(f"\n{'=' * 56}")
    print(f"  结果对比（{NUM_CLASSES} 分类，随机基准 {1 / NUM_CLASSES:.0%}）")
    print(f"{'=' * 56}")
    print(f"  模型A（RNN + MaxPool）          val_acc = {rnn_acc:.4f}")
    print(f"  模型B（直接 Embedding MaxPool）  val_acc = {bow_acc:.4f}")

    # ─── 逐类准确率 ────────────────────────────────────────────────────
    rnn_per_cls = evaluate_per_class(rnn_model, val_loader)
    bow_per_cls = evaluate_per_class(bow_model, val_loader)
    print('\n  各位置准确率（「你」在第 X 位）：')
    print(f"  {'位置':>6}  {'模型A(RNN)':>12}  {'模型B(BoW)':>12}")
    for i in range(NUM_CLASSES):
        print(f"  第 {i + 1} 位   {rnn_per_cls[i]:>10.4f}    {bow_per_cls[i]:>10.4f}")

    # ─── 结论 ──────────────────────────────────────────────────────────
    print(f"""
【结论】
  模型A 使用 RNN 逐字处理，第 t 步的隐藏状态编码了"前 t 字的上下文"，
  因此即使最终对时序做 MaxPool，池化结果仍携带位置信息，能准确判断"你"在哪位。

  模型B 直接对 Embedding 做 MaxPool，等价于词袋(BoW)：
  把句子中所有字符的向量混在一起取最大值，输出与字符顺序完全无关。
  5 个类别对它而言"看起来一模一样"，准确率约等于随机基准 20%。

  → 语序信息对于位置感知类任务至关重要；RNN 通过顺序递推有效保留了这一信息。
""")

    # ─── 推理示例 ──────────────────────────────────────────────────────
    print('─── 推理示例（各模型预测「你」所在位置）──────────────────────')
    rnn_model.eval()
    bow_model.eval()
    random.seed(0)
    print(f"  {'句子':>6}  真实位置  模型A(RNN)  模型B(BoW)")
    with torch.no_grad():
        for pos in range(NUM_CLASSES):
            sent, label = make_sample(pos)
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            rnn_pred = rnn_model(ids).argmax(dim=1).item()
            bow_pred = bow_model(ids).argmax(dim=1).item()
            rnn_mark = '✓' if rnn_pred == label else '✗'
            bow_mark = '✓' if bow_pred == label else '✗'
            print(f"  「{sent}」  第{label + 1}位     "
                  f"{rnn_mark} 第{rnn_pred + 1}位      "
                  f"{bow_mark} 第{bow_pred + 1}位")


if __name__ == '__main__':
    main()
