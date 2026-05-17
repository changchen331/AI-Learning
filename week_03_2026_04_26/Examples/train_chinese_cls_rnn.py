"""
train_chinese_cls_rnn.py
中文句子关键词分类 —— 简单 RNN 版本

任务：句子中含有关键字（好/棒/赞/喜欢/满意）→ 正样本(1)，否则 → 负样本(0)
模型：Embedding → RNN → 取最后隐藏状态 → Linear → Sigmoid
优化：Adam (lr=1e-3)   损失：MSELoss   无需 GPU，CPU 即可运行

依赖：torch >= 2.0   (pip install torch)
"""

import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 4000
MAXLEN = 32
EMBED_DIM = 64
HIDDEN_DIM = 64
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
POS_KEYS = ['好', '棒', '赞', '喜欢', '满意']

TEMPLATES_POS = [
    '这家{}真的很{}，下次还来',
    '这款{}设计让我{}',
    '{}的服务态度让我感到{}',
    '{}体验非常{}',
    '这次购物感觉{}极了',
]

TEMPLATES_NEG = [
    '今天天气阴沉，出门忘带雨伞',
    '这部电影情节比较平淡',
    '下午开了三个小时的会议',
    '路上堵车耽误了不少时间',
    '这道题做了很久还没解出来',
    '最近工作任务比较繁重',
    '超市里人很多，排队结账',
    '这个季节换季容易感冒',
    '今天作业布置得有点多',
    '公交车又晚点了十分钟',
]

OBJ_WORDS = ['店铺', '餐厅', '产品', '服务', '环境', '系统', '设计', '课程']
ADJ_WORDS = ['方便', '简洁', '独特', '舒适', '高效']


def make_positive():
    kw = random.choice(POS_KEYS)
    tmpl = random.choice(TEMPLATES_POS)
    obj = random.choice(OBJ_WORDS)
    try:
        sent = tmpl.format(obj, kw)
    except Exception:
        sent = obj + kw + random.choice(ADJ_WORDS)
    if random.random() < 0.3:
        extra = random.choice(POS_KEYS)
        pos = random.randint(0, len(sent))
        sent = sent[:pos] + extra + sent[pos:]
    return sent


def make_negative():
    base = random.choice(TEMPLATES_NEG)
    if random.random() < 0.4:
        base += random.choice(TEMPLATES_NEG)
    return base


def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n // 2):
        data.append((make_positive(), 1))
        data.append((make_negative(), 0))
    random.shuffle(data)
    return data


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    ids = [vocab.get(ch, 1) for ch in sent]
    ids = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.float),
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class KeywordRNN(nn.Module):
    """
    中文关键词分类器（RNN + MaxPooling 版）
    架构：Embedding → RNN → MaxPool → BN → Dropout → Linear → Sigmoid → (MSELoss)
    """

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len)
        e, _ = self.rnn(self.embedding(x))  # (B, L, hidden_dim)
        pooled = e.max(dim=1)[0]  # (B, hidden_dim)  对序列做 max pooling
        pooled = self.dropout(self.bn(pooled))
        out = torch.sigmoid(self.fc(pooled).squeeze(1))  # (B,)
        return out


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            prob = model(X)
            pred = (prob > 0.5).long()
            correct += (pred == y.long()).sum().item()
            total += len(y)
    return correct / total


def train():
    print("生成数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    model = KeywordRNN(vocab_size=len(vocab))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
        '这款产品真的很棒，非常满意',
        '今天天气有点阴沉，出门带了雨伞',
        '服务太赞了，下次还来',
        '等了很久公交车终于来了',
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            prob = model(ids).item()
            label = '正样本' if prob > 0.5 else '负样本'
            print(f"  [{label}({prob:.2f})]  {sent}")


if __name__ == '__main__':
    train()
