# coding:utf8
import torch

"""
基于pytorch的网络编写
测试dropout层
"""

x = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
dp_layer = torch.nn.Dropout(0.5)
dp_x = dp_layer(x)
print(dp_x)
