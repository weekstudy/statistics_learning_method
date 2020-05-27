# -*-coding:utf-8 -*-
# Author:Jaune
# Date:2020-5-26

'''
k最近邻，就是指最接近的k个邻居（数据），即每个样本都可以由它的K个邻居来表达。
主要思想：在一个含未知样本的空间，可以根据离这个样本最邻近的k个样本的数据类型
来确定样本的数据类型。
3个主要因素：训练集、距离与相似的衡量、k的大小；主要考虑因素：距离与相似度；
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
            # 分割出来的是字符串，转为数字类型
            data_arr.append([int(num) for num in cur_line[1:]])
            label_arr.append(int(cur_line[0]))

    return data_arr, label_arr


def calculate_distance(x1, x2):
    '''
    计算样本值之间的距离，一般有欧式距离，
    闵可夫斯基距离，曼哈顿距离，切比雪夫距离
    :param x1:
    :param x2:
    :return:
    '''
    # 欧式距离
    return np.sqrt(np.sum(np.square(x1 - x2)))
    # 曼哈顿距离
    # return np.sum(np.abs(x1-x2))


def k_nearest_neighour(train_data, train_label, x, top_k):
    '''
    预测样本x的标签
    通过查找k个与x最近的数据点，并统计k个点的标签，
    计算某类标签的个数最多的那类标签
    :param train_data: 训练集
    :param train_label: 训练集的标签
    :param x: 要预测的样本
    :param top_k: 与x最近的k个点
    :return: x的标签
    '''
    # 存储距离
    distance_list = [0] * len(train_data)
    for i in range(len(train_data)):
        x1 = train_data[i]
        # 计算距离
        cur_distance = calculate_distance(x1, x)
        # 存入相应位置
        distance_list[i] = cur_distance
    # argsort 对数组进行从小到大排序，并返回相应索引值的列表
    # 这里取到了距离的前k个索引值
    top_k_list = np.argsort(np.array(distance_list))[:top_k]
    # 用一个数组来存每个类型的次数，
    label_list = [0] * 10
    for index in top_k_list:
        label_list[int(train_label[index])] += 1
    # max(label_list)找到最大的那个数，
    # 返回最大数的索引值，就是标签
    return label_list.index(max(label_list))


def test_model(train_data, train_label, test_data, test_label, topK):
    '''
    计算正确率
    :param train_data: 训练集
    :param train_label: 训练集标签
    :param test_data: 测试集
    :param test_label: 测试集标签
    :param topK: 选择多少个近邻点
    :return: 正确率
    '''
    print("start testing ")
    train_data_mat = np.mat(train_data)
    train_label_mat = np.mat(train_label).T
    test_data_mat = np.mat(test_data)
    test_label_mat = np.mat(test_label).T

    error_counts = 0
    # 这里选择100个点进行测试（渣电脑不能跑测试集的6000个样本）
    test_numbers = 100
    # test_numbers = len(test_label_mat)
    for i in range(test_numbers):
        x = test_data_mat[i]
        predict_y = k_nearest_neighour(train_data_mat, train_label_mat, x, topK)
        if test_label_mat[i] != predict_y:
            print("test start %d/%d-->real lable:%d,predict label:%d" %
                  (i,test_numbers,test_label_mat[i],predict_y))
            error_counts += 1
        else:print('test start %d/%d-->real==predict label:%d' % (i, test_numbers,predict_y))
    return 1-error_counts/test_numbers


if __name__ == "__main__":

    start = time.time()
    # 加载训练集
    train_data, train_label = load_data('./mnist_test/mnist_train.csv')
    # 加载测试集
    test_data, test_label = load_data('./mnist_test/mnist_test.csv')
    # 开始预测，计算正确率
    accuracy_rate = test_model(train_data, train_label, test_data, test_label, 50)
    # 打印正确率
    print("Accuracy Rate is :%.3f" % accuracy_rate)
    end = time.time()
    # 打印用时时间
    print("use time: ", end - start)
