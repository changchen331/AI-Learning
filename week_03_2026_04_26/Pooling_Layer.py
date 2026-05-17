import torch

"""
Pooling Layer
"""

LENGTH = 30


# 最大池化
def max_pooling(xs):
    # 初始化 Pooling
    # max_pool = torch.nn.MaxPool1d(5)  # 入参 5，代表把 5 维池化为 1 维
    last_shape = xs.shape[-1]
    max_pool = torch.nn.MaxPool1d(last_shape)

    # Max Pooling
    y_max = max_pool(xs)
    print("=" * LENGTH, " Max Pooling Layer ", "=" * LENGTH)
    print(f"y_max:\n{y_max.numpy()}")
    print(f"shape: {y_max.shape}")
    y_max_squ = y_max.squeeze()  # 使用 squeeze 去掉值为 1 的维度
    print('-' * LENGTH * 2)
    print(f"y_max_squ:\n{y_max_squ.numpy()}")
    print(f"shape: {y_max_squ.shape}")


# 平均池化
def avg_pooling(xs):
    # 初始化 Pooling
    # avg_pool = torch.nn.AvgPool1d(5) # 入参 5，代表把 5 维池化为 1 维
    avg_pool = torch.nn.AvgPool1d(xs.shape[-1])

    # Avg Pooling
    y_avg = avg_pool(xs)
    print("=" * LENGTH, " Avg Pooling Layer ", "=" * LENGTH)
    print(f"y_avg:\n{y_avg.numpy()}")
    print(f"shape: {y_avg.shape}")
    y_avg_squ = y_avg.squeeze()
    print('-' * LENGTH * 2)
    print(f"y_avg_squ:\n{y_avg_squ.numpy()}")
    print(f"shape: {y_avg_squ.shape}")


if __name__ == '__main__':
    # 生成张量
    Xs = torch.rand([4, 5, 6])
    print("=" * LENGTH, " 张量（未转置） ", "=" * LENGTH)
    print(f"xs:\n{Xs.numpy()}")
    print(f"shape: {Xs.shape}")
    max_pooling(Xs)
    avg_pooling(Xs)

    # 张量转置
    xs_t = Xs.transpose(1, 2)
    print("=" * LENGTH, " 张量（已转置） ", "=" * LENGTH)
    print(f"xs_t:\n{xs_t.numpy()}")
    print(f"shape: {xs_t.shape}")
    max_pooling(xs_t)
    avg_pooling(xs_t)
