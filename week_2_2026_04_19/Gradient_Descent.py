from matplotlib import pyplot

'''
简单模拟梯度下降流程
'''

xs = [0.01 * x for x in range(-250, 251)]
# ys = [1 * x ** 4 + 2 * x ** 3 + 3 * x ** 2 + 4 * x + 5 for x in xs]
ys = [1 * x ** 3 - 5 * x + 4 for x in xs]


# 模型函数
def func(w1, w2, w3, w4, x):
    y = w1 * x ** 3 + w2 * x ** 2 + w3 * x + w4
    return y


# 损失函数（均方误差）
def loss(y_true, y_pred):
    return (y_pred - y_true) ** 2


# 设置超参
lr = 0.001  # 学习率
batch_size = 32  # 批次大小
epochs = 1000  # 训练轮数


def deep_learning(w1=1, w2=1, w3=1, w4=1):
    for epoch in range(epochs):
        epoch_loss = 0
        grad_w1 = 0
        grad_w2 = 0
        grad_w3 = 0
        grad_w4 = 0
        count = 0

        for x, y_true in zip(xs, ys):
            y_pred = func(w1, w2, w3, w4, x)
            epoch_loss += loss(y_true, y_pred)
            count += 1

            # 计算梯度
            grad_w1 += 2 * (y_pred - y_true) * x ** 3
            grad_w2 += 2 * (y_pred - y_true) * x ** 2
            grad_w3 += 2 * (y_pred - y_true) * x
            grad_w4 += 2 * (y_pred - y_true) * 1

            # 梯度下降
            if count == batch_size:
                # 参数更新（SGD）
                w1 -= lr * (grad_w1 / batch_size)
                w2 -= lr * (grad_w2 / batch_size)
                w3 -= lr * (grad_w3 / batch_size)
                w4 -= lr * (grad_w4 / batch_size)

                # 梯度清零
                grad_w1 = 0
                grad_w2 = 0
                grad_w3 = 0
                grad_w4 = 0
                count = 0

        epoch_loss /= len(xs)
        print(f"第 {epoch} 轮: loss:{epoch_loss}\t权重: [w1:{w1}, w2:{w2}, w3:{w3}, w4:{w4}]")
        if epoch_loss < 1e-4:
            break

    # 训练结果展示
    print(f"\n训练后权重: [w1:{w1}, w2:{w2}, w3:{w3}, w4:{w4}]")
    ys_pred = [func(w1, w2, w3, w4, x) for x in xs]  # 使用训练后模型输出预测值

    # 预测值与真实值比对数据分布
    pyplot.scatter(xs, ys, color="red")
    pyplot.scatter(xs, ys_pred, color="blue")
    pyplot.show()


if __name__ == '__main__':
    deep_learning()
