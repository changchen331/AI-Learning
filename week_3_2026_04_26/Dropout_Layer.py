import torch

"""
Dropout Layer
"""

if __name__ == '__main__':
    # 构建输入
    X = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(f"X_init: {X.numpy().tolist()}")

    # Dropout
    dp_layer = torch.nn.Dropout(0.5)
    X_dp = dp_layer(X)
    print(f"X_dp: {X_dp.numpy().tolist()}")
