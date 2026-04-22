import random

import numpy as np
import torch
import torch.nn as nn

'''
手动实现交叉熵
'''


# 进行 softmax 归一化
def softmax(vector):
    return np.exp(vector) / np.sum(np.exp(vector), axis=1, keepdims=True)


# 转化为 one-hot 矩阵
def to_one_hot(vector, shape):
    response = np.zeros(shape)
    for idx, val in enumerate(vector):
        response[idx][val] = 1
    return response


def cross_entropy_loss(logits, labels):
    batch, clas = logits.shape
    logits = softmax(logits)
    labels = to_one_hot(labels, [batch, clas])
    entropy = -1 * np.sum(labels * np.log(logits), 1)
    return sum(entropy) / batch_size


if __name__ == '__main__':
    batch_size = random.randint(3, 10)
    class_num = random.randint(2, 10)

    # 模型预测（模拟）
    predict = []
    for i in range(batch_size):
        temp = []
        for j in range(class_num):
            temp.append(random.uniform(-3, 3))
        predict.append(temp)
    print("模型预测:", predict)
    predict = torch.FloatTensor(predict)

    # 正确结果（模拟）
    target = []
    for i in range(batch_size):
        target.append(random.randint(0, class_num - 1))
    print("正确结果:", target)
    target = torch.LongTensor(target)

    # torch 的交叉熵损失
    ce_loss = nn.CrossEntropyLoss()
    torch_ce_loss = ce_loss(predict, target)
    print(f"\nTorch: {torch_ce_loss}")

    # 我的交叉熵损失
    mine_ce_loss = cross_entropy_loss(predict.numpy(), target.numpy())
    print(f"Mine: {mine_ce_loss}")
