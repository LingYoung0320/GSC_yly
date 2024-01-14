# -*- coding: UTF-8 -*- #
"""
@filename:draw.py
@author:Young
@time:2024-01-13
"""
import matplotlib.pyplot as plt
import numpy as np


# 画图表

def getdata(data_loc):
    epoch_list = []
    # train_loss_list = []
    test_loss_list = []
    acc_list = []
    with open(data_loc, "r") as f:
        for line in f.readlines():
            parts = line.split()
            # 检查是否有足够的部分（每行应该有四个字符串和四个数值）
            if len(parts) == 5:
                # 将每个部分存储到相应的列表中
                epoch_i=(float(parts[1]))
                # train_loss_i=(float(parts[3]))
                test_loss_i=(float(parts[3]))
                acc_i=(float(parts[4]))
            else:
                print(f"不完整的行：{line}")
            print(f"Epoch: {epoch_i}, TestLoss: {test_loss_i}, Accuracy: {acc_i}")
            epoch_list.append(epoch_i)
            # train_loss_list.append(train_loss_i)
            test_loss_list.append(test_loss_i)
            acc_list.append(acc_i)
        print(len(epoch_list), len(acc_list))
        return epoch_list, test_loss_list, acc_list


if __name__ == "__main__":
    data_loc = r"Lstm_MelSpectrogram_50_128_0.01.txt"
    # epoch_list, train_loss_list, test_loss_list, acc_list = getdata(data_loc)
    epoch_list, test_loss_list, acc_list = getdata(data_loc)

    # train_loss
    # plt.plot(epoch_list, train_loss_list)
    #
    # plt.legend(["model"])
    # plt.xticks(np.arange(0, 11, 1))  # 横坐标的值和步长
    # plt.yticks(np.arange(0, 4, 1))  # 纵坐标的值和步长
    # plt.xlabel("Epoch")
    # plt.ylabel("train_loss")
    # plt.title("Train Loss")
    # plt.show()

    # 准确率曲线
    plt.plot(epoch_list, acc_list)

    plt.legend(["model"])
    plt.xticks(np.arange(0, 51, 10))  # 横坐标的值和步长
    plt.yticks(np.arange(0, 101, 10))  # 纵坐标的值和步长
    plt.xlabel("Epoch")
    plt.ylabel("Accurancy(100%)")
    plt.title("Model Accuracy")
    plt.show()

    # test_loss
    # plt.plot(epoch_list, test_loss_list)
    #
    # plt.legend(["model"])
    # plt.xticks(np.arange(0, 11, 1))  # 横坐标的值和步长 表示0-10 0：起始点 11：终止点，不包括在内
    # plt.yticks(np.arange(0, 0.0006, 0.0001))  # 纵坐标的值和步长
    # plt.xlabel("Epoch")
    # plt.ylabel("test_loss(100%)")
    # plt.title("Test Loss")
    # plt.show()