import random

import torch
import torch.nn as nn

'''
模拟一个 3 层神经网络模型
'''


class TorchModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=5, output_size=2):
        super(TorchModel, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x0):
        # 定义前向传播流程
        x1 = self.fc1(x0)
        x2 = self.fc2(x1)
        y = self.fc3(x2)
        return y


class MineModel:
    def __init__(self, w1, w2, w3, b1, b2, b3):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

    def forward(self, x0):
        # 前向传播计算
        x1 = (x0 @ self.w1.T) + self.b1  # (x0 · w1_T) + b1 = x1
        x2 = (x1 @ self.w2.T) + self.b2  # (x1 · w2_T) + b2 = x2
        y = (x2 @ self.w3.T) + self.b3  # (x2 · w3_T) + b3 = y
        return y


if __name__ == '__main__':
    size_input = random.randint(3, 10)
    size_hidden = random.randint(5, 10)
    size_output = random.randint(2, 10)
    length = 150

    # 模型输入（模拟）
    x = []
    size = random.randint(1, 10)
    for i in range(size):
        temp = []
        for j in range(size_input):
            temp.append(random.uniform(-3, 3))
        x.append(temp)
    print("模型输入: ", x)
    print("=" * length)
    x = torch.FloatTensor(x)

    # Torch 的神经网络模型
    torch_model = TorchModel(size_input, size_hidden, size_output)
    print(torch_model.state_dict())
    print("=" * length)

    # 获取 Torch 模型的 W（权重矩阵）和 b（偏置向量）
    torch_model_w1 = torch_model.state_dict()["fc1.weight"].numpy()
    torch_model_b1 = torch_model.state_dict()["fc1.bias"].numpy()
    print("Torch w1: ", torch_model_w1)
    print("Torch b1: ", torch_model_b1)
    print("-" * length)

    torch_model_w2 = torch_model.state_dict()["fc2.weight"].numpy()
    torch_model_b2 = torch_model.state_dict()["fc2.bias"].numpy()
    print("Torch w2: ", torch_model_w2)
    print("Torch b2: ", torch_model_b2)
    print("-" * length)

    torch_model_w3 = torch_model.state_dict()["fc3.weight"].numpy()
    torch_model_b3 = torch_model.state_dict()["fc3.bias"].numpy()
    print("Torch w3: ", torch_model_w3)
    print("Torch b3: ", torch_model_b3)
    print("=" * length)

    # 使用 Torch 模型进行预测
    torch_predict = torch_model.forward(x)
    print("Torch predict: ", torch_predict)
    print("-" * length)

    # 使用 Torch 的初始权重（W）和偏置（b）自行计算
    mine_model = MineModel(torch_model_w1, torch_model_w2, torch_model_w3,
                           torch_model_b1, torch_model_b2, torch_model_b3)
    mine_predict = mine_model.forward(x.numpy())
    mine_predict_print = [[f"{x:.4f}" for x in line] for line in mine_predict]
    print("Mine predict: ", mine_predict_print)
