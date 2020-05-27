# -*-coding:utf-8 -*-
# Author:Jaune
# Date:2020-5-27

'''
朴素贝叶斯主要假设：特征之间条件独立，
先基于训练集学习x与y的联合概率分布，然后
对于给定的样本利用贝叶斯定理求出最大后验概率的y
'''

import numpy as np
import time


def load_data(filename):
    '''
    加载数据
    :param filename:文件路径
    :return: 数据集和标签
    '''
    print("start loading data")
    data_arr = []
    label_arr = []
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            cur_line = line.strip().split(',')
            # 分割出来的是字符串，转为数字类型,
            # 为了简化计算，将特征值转为二值，0-1
            data_arr.append([int(int(num) > 128) for num in cur_line[1:]])
            label_arr.append(int(cur_line[0]))

    return data_arr, label_arr


def calc_probability(train_data, train_label):
    '''
    通过训练集计算先验概率和条件概率分布
    :param train_data: 训练集
    :param train_label: 训练集标签
    :return: 先验概率和条件概率值
    '''
    print("start training")

    feature_nums = 784
    class_nums = 10
    # 存放先验概率，p{y=0}先验概率值放在pri_prob_y[0]，依次类推
    pri_prob_y = np.zeros((class_nums, 1))

    for i in range(class_nums):
        # 统计某类标签出现的次数
        counts = np.sum(np.mat(train_label) == i)
        # 为了防止分子分母为零，分子加lambda>0,分母加lambda*k,k为分类数，对结果无影响
        pri_prob_y[i] = (counts + 1) / (len(train_data) + 10)

    # 概率值太小，为了防止下溢，取对数计算
    pri_prob_y = np.log(pri_prob_y)

    # 计算条件概率值,用3维数组存放
    # 因为有10个类别，每个类别有784维特征，每个特征取值是0或者是1，
    con_prob_xy = np.zeros((class_nums, feature_nums, 2))

    # 统计每个类别，每维特征为0或者1的样本数
    for i in range(len(train_label)):
        cur_label = train_label[i]
        cur_x = train_data[i]

        for j in range(feature_nums):
            # 比如0标签下，第一维为0的样本个数，
            # 样本的第j维，cur_x[j]表示当前样本第j维的特征值(0 or 1)
            # 比如cur_x[j]=0，就在[0][j][0]加1
            con_prob_xy[cur_label][j][cur_x[j]] += 1
    res = con_prob_xy[0]

    for label in range(class_nums):
        for j in range(feature_nums):
            # 在给定label下，第j维特征取值为0的个数
            prob_xy0 = con_prob_xy[label][j][0]
            # 在给定label下，第j维特征取值为1的个数
            prob_xy1 = con_prob_xy[label][j][1]
            # 为了防止分子分母为零，分子加lambda>0,
            # 分母加lambda*s,s为每个特征的取值个数，
            # 分母+2是因为特征值有0/1两种取值
            # 条件概率
            con_prob_xy[label][j][0] = np.log((prob_xy0 + 1) / (prob_xy0 + prob_xy1 + 2))
            con_prob_xy[label][j][1] = np.log((prob_xy1 + 1) / (prob_xy0 + prob_xy1 + 2))

    return pri_prob_y, con_prob_xy


def naive_bayes(prob_y, prob_xy, x):
    '''
    利用贝叶斯定理求后验概率
    :param prob_y: 先验概率分布
    :param prob_xy: 条件概率分布
    :param x: 要估计的样本
    :return: 所有标签的概率
    '''
    featureNums = 784
    classNums = 10
    # 存放每个类别估计的概率值
    prob_list = [0] * classNums
    for i in range(classNums):
        prob_sum = 0
        # 可以理解为在给定标签的情况下，每一维取值的概率，
        # 比如con_prob_xy[0][0][x[0]]表示在标签为0的情况下，第1维取值为x[0]的概率
        for j in range(featureNums):
            # 因为在计算概率时取了对数，因此变为相加
            # 这里相当于p(X=x|y)=p(x1|y)*p(x2|y)...p(xn|y)，书上4.1.1公式4.3
            prob_sum += prob_xy[i][j][x[j]]
        prob_list[i] = prob_sum + prob_y[i]

    # 返回最大后验概率值的索引值，即标签
    return prob_list.index(max(prob_list))


def test_mdoel(prob_y, prob_xy, test_data, test_label):
    '''
    测试集测试
    :param prob_y:先验概率分布
    :param prob_xy: 条件概率分布
    :param test_data: 测试集数据
    :param test_label: 测试集标签
    :return: 准确率
    '''
    error_count = 0
    for i in range(len(test_data)):
        predict_y = naive_bayes(prob_y, prob_xy, test_data[i])
        if predict_y != test_label[i]:
            error_count += 1

    return 1 - error_count / len(test_data)


if __name__ == "__main__":
    start = time.time()

    # 加载训练集
    train_data, train_label = load_data("./mnist_test/mnist_train.csv")
    # 加载测试集
    test_data, test_label = load_data("./mnist_test/mnist_test.csv")

    # 计算先验概率，条件概率
    prob_y, prob_xy = calc_probability(train_data, train_label)
    # 开始测试
    accuracy = test_mdoel(prob_y, prob_xy, test_data, test_label)
    print("the accuracy rate is :", accuracy)

    end = time.time()

    print("time:", end - start)
