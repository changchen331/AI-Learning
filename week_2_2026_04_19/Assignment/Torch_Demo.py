import random

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

"""
多分类任务的训练: 一个随机向量，哪一维数字最大就属于第几类
"""

length = 30


class TorchModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TorchModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ac1 = torch.nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.ac2 = nn.Softmax(dim=1)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x0):
        x1_1 = self.fc1(x0)  # (batch * input) -> (batch * hidden)
        x1_2 = self.ac1(x1_1)  # (batch * hidden) -> (batch * hidden)
        x2_1 = self.fc2(x1_2)  # (batch * hidden) -> (batch * output)
        x2_2 = self.ac2(x2_1)
        y_pred = x2_2
        return y_pred

    def loss(self, x0, y_true):
        x1_1 = self.fc1(x0)  # (batch * input) -> (batch * hidden)
        x1_2 = self.ac1(x1_1)  # (batch * hidden) -> (batch * hidden)
        x2_1 = self.fc2(x1_2)  # (batch * hidden) -> (batch * output)
        y_pred = x2_1
        return self.ce_loss(y_pred, y_true)


# 构建数据集
def build_dataset(dataset_size, input_dim):
    xs = []
    for i in range(dataset_size):
        temp = np.random.uniform(0, 1, input_dim * 10)
        selected = random.sample(list(temp), input_dim)
        xs.append(selected)
    ys = np.argmax(xs, axis=1)
    return torch.FloatTensor(xs), torch.LongTensor(ys)


# 模型训练
def train_model(model, train_xs, train_ys, valid_xs, valid_ys, print_or_not=False):
    # 设置超参数
    epoch_num = 1000
    batch_size = 5
    learning_rate = 0.001
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    accuracies = []
    losses = []
    # 训练模型
    for epoch in range(epoch_num):
        model.train()
        loss_list = []
        for batch_index in range(len(train_xs) // batch_size):
            # 获取当前 Batch 数据
            x = train_xs[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_ys[batch_index * batch_size:(batch_index + 1) * batch_size]
            # 计算 loss
            loss = model.loss(x, y)  # 计算交叉熵损失
            loss_list.append(loss.item())
            # 梯度下降
            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 梯度计算
            optimizer.step()  # 更新参数
        mean_loss = sum(loss_list) / len(loss_list)
        print("-" * length, f" 第 {epoch + 1} 轮平均 loss: {mean_loss} ", "-" * length)
        accuracy = evaluate_model(model, valid_xs, valid_ys)
        # 日志记录
        accuracies.append(accuracy)
        accuracy_log = f"Acc: {accuracy * 100:.2f}%"
        losses.append(mean_loss)
        mean_loss_log = f"Loss: {mean_loss:.4f}"
        log.append([accuracy_log, mean_loss_log])
        if mean_loss < 5 * 1e-3:
            break
    # 训练完成
    torch.save(model.state_dict(), "./model.pth")  # 保存模型
    print("=" * 4 * length)
    # 画图（可选）
    if print_or_not:
        plt.plot(range(len(log)), [acc for acc in accuracies], label="Accuracy")
        plt.plot(range(len(log)), [loss for loss in losses], label="Loss")
        plt.legend()
        plt.savefig("./acc-loss.png")
        plt.show()
    return log


# 评估
def evaluate_model(model, xs, ys):
    model.eval()
    correct = 0
    with torch.no_grad():  # 关闭自动求导机制（Autograd），禁止计算梯度
        y_pred = model.forward(xs)
        for y_p, y_t in zip(y_pred, ys):
            pred = y_p.argmax(dim=0)
            true = y_t
            if pred == true:
                correct += 1  # 分类正确
    accuracy = correct / len(xs)
    print(f"正确分类个数: {correct}/{len(ys)}个 - 正确率: {accuracy * 100:.2f}%")
    return accuracy


def predict(model, xs):
    model.eval()
    with torch.no_grad():
        y_pred = model.forward(xs)
    for y_p, x in zip(y_pred, xs.numpy()):
        prob = [f"{y * 100:.2f}%" for y in y_p]
        pred = y_p.argmax(dim=0)
        print(f"输入: {x} - 概率: {prob} - 预测: {pred}")


if __name__ == '__main__':
    # 向量维度
    input_size = 5
    hidden_size = 5
    output_size = 5

    # 数据集规模
    train_size = 1000
    valid_size = 100
    test_size = 10

    mode = 1
    if mode == 0:
        # 构建训练集
        Train_xs, Train_ys = build_dataset(train_size, input_size)
        print(f"训练集构建完成 ({len(Train_xs)}, {len(Train_ys)})")
        # 构建验证集
        Valid_xs, Valid_ys = build_dataset(valid_size, input_size)
        print(f"验证集构建完成 ({len(Valid_xs)}, {len(Valid_ys)})")

        # 训练 + 验证 模型
        new_model = TorchModel(input_size, hidden_size, output_size)
        training_log = train_model(new_model, Train_xs, Train_ys, Valid_xs, Valid_ys, True)
        print("模型训练完毕")
        print("=" * 4 * length)
    elif mode == 1:
        # 构建测试集
        Test_xs, Test_ys = build_dataset(test_size, input_size)
        print(f"测试集构建完成 ({len(Test_xs)}, {len(Test_ys)})")
        print("=" * 4 * length)

        # 测试模型
        mine_model = TorchModel(input_size, hidden_size, output_size)
        mine_model.load_state_dict(torch.load("./model.pth"))
        predict(mine_model, Test_xs)
        evaluate_model(mine_model, Test_xs, Test_ys)
