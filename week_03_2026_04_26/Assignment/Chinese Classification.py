import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

"""
中文分类任务
对一个任意包含“原神”字的五个字的文本，“原神”在第几位，就属于第几类
"""

# 初始化
SEED = 31
random.seed(SEED)
torch.manual_seed(SEED)

LENGTH = 50

TARGET = "原神"  # TARGET 在哪就是第几类

# 超参数
MAX_LENGTH = 5  # Token 最大长度
DATA_SIZE = 1000  # 总数据量
TRAIN_RATIO = 0.8  # 训练数据占比
VALID_RATIO = 0.1  # 验证数据占比
BATCH_SIZE = 32
EMBED_DIM = 64
HIDDEN_DIM = 64
OUTPUT_DIM = 5
DROPOUT = 0.3
EPOCH = 100
LR = 1e-3


# 构建词表
def build_vocab():
    vocab = {'[PAD]': 0, TARGET: 1}
    size = min(10000, DATA_SIZE * MAX_LENGTH * 2)
    samples = random.sample(range(0x4e00, 0x9fff), size)  # 随机中文字符
    for sample in samples:
        vocab[chr(sample)] = len(vocab)
    vocab['[UNK]'] = len(vocab)
    return vocab


# 构建数据
def build_data(vocab, size=DATA_SIZE):
    data = []
    words = list(vocab.keys())
    for i in range(size):
        temp = [TARGET]
        for j in range(MAX_LENGTH - 1):
            position = random.randint(2, len(words) - 1)
            temp.append(words[position])
        random.shuffle(temp)
        data.append(temp)
    return data


# Encode Sentence
def encode(words, vocab, max_len=MAX_LENGTH):
    encoded_data = [vocab.get(word, vocab['[UNK]']) for word in words]
    encoded_data = encoded_data[:max_len]
    encoded_data += [vocab['[PAD]']] * (max_len - len(encoded_data))
    return encoded_data


# DataLoader
class MyDataset(Dataset):
    def __init__(self, data, vocab):
        self.x = [encode(datum, vocab) for datum in data]
        self.y = [datum.index(TARGET) for datum in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return (
            torch.tensor(self.x[item], dtype=torch.long),
            torch.tensor(self.y[item], dtype=torch.long)
        )


# 构建模型
class CnClassRNN(nn.Module):
    """
    中文关键词分类器（RNN + MaxPooling 版）
    架构：Embedding → RNN → MaxPool → BN → Dropout → Linear → Softmax → (Cross Entropy)
    """

    def __init__(self, in_dim, emb_dim=EMBED_DIM, hid_dim=HIDDEN_DIM, out_dim=OUTPUT_DIM, dp=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(in_dim, emb_dim, padding_idx=0)
        self.rnn = nn.RNN(emb_dim, hid_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hid_dim)
        self.dropout = nn.Dropout(dp)
        self.fc = nn.Linear(hid_dim, out_dim)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        embedding = self.embedding(x)  # (B,L,I)->(B,L,E)
        rnn, _ = self.rnn(embedding)  # (B,L,E)->(B,L,H)
        max_pool = rnn.max(dim=1)[0]  # (B,L,H)->(B,H)
        bn = self.bn(max_pool)  # (B,H)->(B,H)
        dropout = self.dropout(bn)  # (B,H)->(B,H)
        fc = self.fc(dropout)  # (B,H)->(B,O)
        logits = fc
        return logits

    def loss(self, x, y):
        logits = self.forward(x)
        loss = self.ce_loss(logits, y)
        return loss


# 训练模型
def train(model, train_loader, valid_loader):
    # 训练初始化
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}\n")

    # 模型训练
    for ep in range(0, EPOCH):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            # 计算损失
            loss = model.loss(x, y)  # 交叉熵损失
            # 梯度下降
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, valid_loader)
        print(f"Epoch {(ep + 1):2d}/{EPOCH}: Loss={avg_loss:.4f} - Val_Acc={val_acc * 100:.2f}%")
        print('-' * LENGTH * 2)
        if avg_loss < 5 * 1e-3:
            break
    # 训练完成
    torch.save(model.state_dict(), "model.pth")  # 保存模型
    print(f"最终验证准确率：{evaluate(model, valid_loader):.4f}")


# 计算分数
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            logits = torch.softmax(model(x), dim=1)
            for y_p, y_t in zip(logits, y):
                pred = y_p.argmax(dim=0)
                true = y_t
                if pred == true:
                    correct += 1
                total += 1
    accuracy = correct / total
    # print(f"正确分类个数: {correct}/{total}个 - 正确率: {accuracy * 100:.2f}%")
    return accuracy


# 模型测试
def test(model, data_loader):
    model.eval()
    with torch.no_grad():
        for xs, ys in data_loader:
            yps = torch.softmax(model(xs), dim=1)
            for x, yp, y in zip(xs, yps, ys):
                prob = [f"{y * 100:.2f}%" for y in yp]
                pred = yp.argmax(dim=0)
                print(f"输入: {x.detach().numpy()} - 概率: {prob} - "
                      f"预测: {pred} - 答案: {y.detach().numpy()}")


if __name__ == '__main__':
    # 构建词表
    print("=" * LENGTH, " 生成数据 ", "=" * LENGTH)
    Vocab = build_vocab()
    Data = build_data(Vocab)
    print(f"样本数: {len(Data)}, 词表大小: {len(Vocab)}")

    # 构建数据集
    print("=" * LENGTH, " 构建数据集 ", "=" * LENGTH)
    split_train = int(len(Data) * TRAIN_RATIO)
    split_valid = split_train + int(len(Data) * VALID_RATIO)
    split_test = split_train + split_valid

    mode = 1
    if mode == 0:
        # 构建训练集
        train_data = Data[:split_train]
        train_dataset = MyDataset(train_data, Vocab)
        Train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
        print(f"Train x:\n{train_dataset.x}")
        print(f"Train y:\n{train_dataset.y}")
        print(f"Train size: {len(train_dataset)}")
        print("-" * LENGTH * 2)
        # 构建验证集
        valid_data = Data[split_train:split_valid]
        valid_dataset = MyDataset(valid_data, Vocab)
        Valid_loader = DataLoader(valid_dataset, BATCH_SIZE, shuffle=True)
        print(f"Valid x:\n{valid_dataset.x}")
        print(f"Valid y:\n{valid_dataset.y}")
        print(f"Valid size: {len(valid_dataset)}")
        print("-" * LENGTH * 2)
        # 模型初始化
        Mine_model = CnClassRNN(len(Vocab))
        # 模型训练
        print("=" * LENGTH, " 模型训练 ", "=" * LENGTH)
        train(Mine_model, Train_loader, Valid_loader)
    else:
        # 构建测试集
        test_data = Data[split_valid:]
        test_dataset = MyDataset(test_data, Vocab)
        Test_loader = DataLoader(test_dataset, BATCH_SIZE)
        print(f"Test x:\n{test_dataset.x}")
        print(f"Test y:\n{test_dataset.y}")
        print(f"Test size: {len(test_dataset)}")
        # 模型初始化
        Mine_model = CnClassRNN(len(Vocab))
        Mine_model.load_state_dict(torch.load("model.pth"))
        # 模型测试
        print("=" * LENGTH, " 模型测试 ", "=" * LENGTH)
        test(Mine_model, Test_loader)
        acc = evaluate(Mine_model, Test_loader)
        print(f"正确率: {acc * 100:.2f}%")
