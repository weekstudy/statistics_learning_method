# -*-coding:utf-8 -*-
# Author:Jaune
# Date:2020-5-26
'''
前提：数据必须是线性可分的
主要思想：根据分类的错误点来不断调整超平面的位置
'''

import numpy as np
import time


def load_data(fileName):
    '''
    加载数据集Mnist
    :param fileName:数据集路径
    :return: list形式的数据集及标签
    '''
    print("start loading data")
    data_arr = []
    label_arr = []
    # fr = open(fileName,'r')
    with open(fileName, 'r') as fr:
        for line in fr.readlines():
            # strip()函数去除头尾指定字符
            # split()函数按指定字符分割
            cur_line = line.strip().split(',')

            # 因为是二分类问题，因此将>=5的分为一类，标签为1
            # 拓展：是否可以将相似的数字分为一类，比如1/4/7一类
            if int(cur_line[0]) >= 5:
                label_arr.append(1)
            else:
                label_arr.append(-1)
            # 将每行数值进行归一化
            data_arr.append([int(num) / 255 for num in cur_line[1:]])

    return data_arr, label_arr


def perceptron(data_arr, label_arr, iter=50):
    '''
    感知机训练
    :param data_arr: 训练集，二维数组，一行代表一个数字
    :param label_arr: 标签，1表示>=5的数字
    :param iter: 迭代次数，默认50次
    :return:w,b
    '''
    # 转成矩阵形式,便于计算,
    # 矩阵跟数组的取值方式不一样，矩阵[m,n],二维数组[m][n]
    data_mat = np.mat(data_arr)
    label_mat = np.mat(label_arr).T
    # m为样本个数，n为特征值个数
    m, n = np.shape(data_mat)
    # 初始化权重和偏置值,学习率
    w = np.zeros((1, n))
    b, lr = 0, 0.0001

    # 开始迭代
    for i in range(iter):
        # 可以用随机梯度下降和梯度下降
        # 随机梯度下降就是随机选取一个样本的梯度值进行迭代，
        # 梯度下降是用全部样本的梯度值进行迭代,常见的是随机梯度下降
        # 其实这里不是随机梯度下降，没有随机选择样本
        for j in range(m):
            xi = data_mat[i]
            yi = label_mat[i]
            if yi * (w * xi.T + b) <= 0:
                w = w + lr * yi * xi
                b = b + lr * yi
        print("Round %d/%d training" % (i, iter))

    return w, b


def test_model(test_data, test_label, w, b):
    '''
    测试准确率
    :param test_data: 测试集
    :param label_data: 标签
    :param w: 权重
    :param b: 偏置值
    :return: 准备率
    '''
    print("start testing")
    data_mat = np.mat(test_data)
    label_mat = np.mat(test_label).T
    m, n = np.shape(data_mat)
    err_counts = 0
    for i in range(m):
        xi = data_mat[i]
        yi = label_mat[i]
        # 分类结果
        result = yi * (w * xi.T + b)
        # 如果<=0，则分类错误
        if result <= 0:
            err_counts += 1
    # 计算正确率
    acc_rate = 1 - err_counts / m
    return acc_rate


if __name__ == "__main__":
    start = time.time()
    # 获取训练集
    train_data, train_label = load_data('./mnist_test/mnist_train.csv')
    # 获取测试集
    test_data, test_label = load_data('./mnist_test/mnist_test.csv')
    # 计算参数值
    w, b = perceptron(train_data, train_label, iter=30)
    # 测试正确率
    accuracy_rate = test_model(test_data, test_label, w, b)

    end = time.time()
    print("accuracy rate is :", accuracy_rate)
    print("times: ", end - start)
