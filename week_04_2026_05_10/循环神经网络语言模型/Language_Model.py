import argparse
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

"""
字符级语言模型训练脚本，支持 RNN / LSTM 切换，含 PPL 计算
"""

LENGTH = 30


def load_data(data_path):
    texts = []
    for filepath in Path(data_path).glob("*.txt"):
        with open(filepath, encoding="utf-8") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    chars = sorted(set(text))
    char2id = dict(zip(chars, range(len(chars))))
    id2char = dict(zip(range(len(chars)), chars))
    return char2id, id2char


class CharDataset(Dataset):
    def __init__(self, text, char2id, seq_length):
        ids = [char2id[c] for c in text if c in char2id]
        self.data = torch.tensor(ids, dtype=torch.long)
        self.seq_length = seq_length

    def __len__(self):
        # 返回要预测 token 的数量
        return max(0, (len(self.data) - self.seq_length))

    def __getitem__(self, idx):
        # 给出 (idx, idx+len)
        x = self.data[idx:idx + self.seq_length]
        # 预测 (idx+1, idx+len+1)
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y


class NnLm(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size,
                 num_layer, model_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        model_types = {
            "lstm": nn.LSTM,
            "rnn": nn.RNN,
        }
        model = model_types.get(model_type, nn.LSTM)
        self.layer = model(
            embedding_dim,
            hidden_dim,
            num_layers=num_layer,
            batch_first=True,
            # 只在多层 RNN 时启用层间 Dropout，单层设为 0
            dropout=dropout if num_layer > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.embedding(x)  # (B, S) -> (B, S, E)
        output, _ = self.layer(e)  # (B, S, E) -> (B, S, H)
        output = self.dropout(output)  # (B, S, H)
        logits = self.fc(output)  # (B, S, H) -> (B, S, V)
        return logits


def train(model, train_data, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for x, y in train_data:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        # 为什么要变形？
        # 因为 CrossEntropyLoss 期望输入形状为 (N, C)，其中 N 是样本数量，C 是类别数量。
        # 而 logits 形状是 (B, S, V)，需要将其变为 (B*S, V)，同时将 y 变为 (B*S,)。
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 权重更新

        # 总损失 = 平均损失 × 有效 token 数
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()  # 有效 token 数

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


@torch.no_grad()
def evaluate(model, data, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x, y in data:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)

        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm", choices=["rnn", "lstm"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--corpus", default="../data")
    parser.add_argument("--save", default="best_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} model: {args.model.upper()}")

    # 准备数据
    data = load_data(args.corpus)
    if data is None:
        raise FileNotFoundError("No data")
    print(f"Loaded {len(data)} characters")

    char2id, id2char = build_vocab(data)
    vocab_size = len(char2id)
    print(f"Vocabulary size: {vocab_size}")

    lines = data.splitlines()
    random.shuffle(lines)
    valid_ratio = 0.2
    test_ratio = 0.1
    valid_split = int(len(lines) * valid_ratio)
    test_split = int(len(lines) * test_ratio)
    valid_data = "\n".join(lines[:valid_split])
    test_data = "\n".join(lines[valid_split:valid_split + test_split])
    train_data = "\n".join(lines[valid_split + test_split:])

    train_dataset = CharDataset(train_data, char2id, args.seq_len)
    valid_dataset = CharDataset(valid_data, char2id, args.seq_len)
    test_dataset = CharDataset(test_data, char2id, args.seq_len)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False, num_workers=4, drop_last=True)

    # 模型
    model = NnLm(
        embedding_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        vocab_size=vocab_size,
        num_layer=args.num_layers,
        model_type=args.model,
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_valid_ppl = float("inf")
    print("\nStart training...")
    print("=" * LENGTH * 2)

    for epoch in range(args.epochs):
        train_loss, train_ppl = train(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_ppl = evaluate(model, valid_loader, criterion, device)

        mark = "  *" if valid_ppl < best_valid_ppl else ""
        if valid_ppl < best_valid_ppl:
            best_valid_ppl = valid_ppl
            torch.save({
                "state_dict": model.state_dict(),
                "char2id": char2id,
                "id2char": id2char,
                "args": vars(args),
            }, args.save)

        print(f"Epoch {epoch + 1:02d} | Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f} | "
              f"Valid Loss: {valid_loss:.4f} | Valid PPL: {valid_ppl:.2f} {mark}")

    print("\nTraining complete.")
    print(f"Best Valid PPL: {best_valid_ppl:.2f} Saved to {args.save}")


if __name__ == '__main__':
    main()
