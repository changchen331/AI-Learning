import numpy as np
import torch
from torch import nn

"""
手动实现 Adam 优化器
"""

length = 30

# 超参数
Learning_Rate = 0.001
Beta1 = 0.9
Beta2 = 0.999
Epsilon = 1e-8


# 简单网络（一层线性层）
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.mse = nn.MSELoss()

    def forward(self, x):
        return self.fc1(x)

    def loss(self, pred, y):
        return self.mse(pred, y)


# Torch Adam
def torch_adam(model, x, y):
    w_init = model.fc1.weight.data.clone()  # 初始权重
    # Adam 优化器（Torch）
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate,
                                 betas=(Beta1, Beta2), eps=Epsilon)

    # 计算损失
    pred = model(x)
    loss = model.loss(pred, y)

    # 梯度下降
    optimizer.zero_grad()  # 梯度归零
    loss.backward()  # 反向传播（计算梯度）
    grad_torch = model.fc1.weight.grad.clone()  # 获取梯度
    optimizer.step()  # 更新参数
    w_after_torch = model.fc1.weight.data.clone()  # 更新后参数

    # 展示数据
    print("=" * length, " Torch Adam ", "=" * length)
    print(f"Loss: {loss.item():.4f}")
    print(f"Weight(init):\n{w_init.numpy()}")
    print(f"Gradient:\n{grad_torch.numpy()}")
    print(f"Weight(after):\n{w_after_torch.numpy()}")

    return w_after_torch.numpy(), grad_torch


# Mine Adam
def mine_adam(w_init, grad):
    # 获取初始权重和梯度
    w = w_init.numpy().copy()
    g = grad.numpy().copy()

    # 初始化（t=0，mt,vt 均为零）
    t = 0
    mt = np.zeros_like(w)
    vt = np.zeros_like(w)

    # Adam 优化
    t += 1
    gt = g
    # 更新矩阵
    mt = Beta1 * mt + (1 - Beta1) * gt  # 更新一阶矩
    vt = Beta2 * vt + (1 - Beta2) * gt ** 2  # 更新二阶矩
    # 偏差修正
    mth = mt / (1 - Beta1 ** t)  # mt 偏差修正
    vth = vt / (1 - Beta2 ** t)  # vt 偏差修正
    # 参数更新
    w_after_mine = w_init - Learning_Rate * mth / (np.sqrt(vth) + Epsilon)

    # 展示数据
    print("=" * length, " Mine Adam ", "=" * length)
    print(f"mt:\n{mt}")
    print(f"vt:\n{vt}")
    print(f"mth:\n{mth}")
    print(f"vth:\n{vth}")
    print(f"Weight(init):\n{w_init.numpy()}")
    print(f"Gradient:\n{grad.numpy()}")
    print(f"Weight(after):\n{w_after_mine.numpy()}")

    return w_after_mine


if __name__ == '__main__':
    # 构建数据集
    size = np.random.randint(4, 8)
    input_size = np.random.randint(5, 10)
    output_size = np.random.randint(1, 5)
    X = torch.rand(size, input_size)
    Y = torch.rand(size, output_size)
    print("=" * length, " Data ", "=" * length)
    print(f"X: \n{X.numpy()}")
    print(f"Y: \n{Y.numpy()}")

    # 初始化
    Mine_model = Model(input_size, output_size)
    W_init = Mine_model.fc1.weight.data.clone()  # 初始权重矩阵

    # Torch Adam
    W_after_torch, Grad = torch_adam(Mine_model, X, Y)

    # Mine Adam
    W_after_mine = mine_adam(W_init, Grad)

    # 对比
    diff = np.abs(W_after_mine - W_after_mine)
    print("=" * length, " Adam: Torch vs. Mine ", "=" * length)
    print(f"最大误差: {diff.max():.2e}")
    print(f"平均误差: {diff.min():.2e}")
