import numpy as np
import torch
from torch import nn

"""
手动实现 Batch Normalization
"""

LENGTH = 30


# Torch Batch Normalization
def torch_bn(x):
    bn = nn.BatchNorm1d(x.shape[-1])
    y = bn(x)
    print("=" * LENGTH, " Torch Batch Normalization ", "=" * LENGTH)
    print(f"Torch Batch Norm:\n{y.detach().numpy()}")
    return bn


# Mine Batch Normalization
def mine_bn(bn, x):
    # 获取参数（γ、β）
    gamma = bn.state_dict()["weight"].numpy()
    beta = bn.state_dict()["bias"].numpy()

    # 超参数
    epsilon = 1e-5

    # 计算 Mean & Variance
    mean = np.mean(x, axis=0)  # 平均值
    vari = np.var(x, axis=0)  # 方差
    # 归一化
    x_norm = (x - mean) / np.sqrt(vari + epsilon)
    # 仿射变换
    y = gamma * x_norm + beta

    print("=" * LENGTH, " Mine Batch Normalization ", "=" * LENGTH)
    print(f"Mine Batch Norm:\n{y}")


if __name__ == '__main__':
    m = np.random.randint(3, 10)
    n = np.random.randint(3, 5)
    xs = np.random.random((m, n))
    print("=" * LENGTH, " 输入 ", "=" * LENGTH)
    print(f"xs: {xs}")
    print(f"shape: {xs.shape}")

    BN = torch_bn(torch.FloatTensor(xs))
    mine_bn(BN, xs)
