# -*-coding:utf-8 -*-
# Author:Jaune
# Date:2020-5-27

'''
Logistic regression
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
            data_arr.append([int(num) / 255 for num in cur_line[1:]])
            # 为了简化，就做二分类，将标签为0的标为1，其他都为1，
            # 其实也可以做多分类
            if int(cur_line[0]) == 0:
                label_arr.append(1)
            else:
                label_arr.append(0)

    # return np.array(data_arr), np.array(label_arr)
    return data_arr, label_arr


def predict(weight,x):
    '''
    预测
    :param weight:权重
    :param x: 预测样本
    :return: 预测结果
    '''
    # 点积
    w_x=np.dot(weight,x)
    exp_val = np.exp(w_x)
    prob = exp_val/(1+exp_val)
    if prob>=0.5:
        return 1
    return 0


def logistic_regression(train_data,train_label,iter=200):
    '''
    计算参数值，W，b
    :param train_data:训练集
    :param train_label: 标签
    :param iter: 迭代次数
    :return: 返回权重
    '''
    # 这里将b与w合在一起，x多一维,值为1
    for i in range(len(train_data)):
        train_data[i].append(1)

    # 转为二维数组
    train_data=np.array(train_data)
    # 初始化权重,0
    weight = np.zeros(train_data.shape[1])
    # 设置学习率
    lr=0.001
    #开始迭代,采用梯度上升，批量梯度上升每一次迭代
    # 需要计算所有样本的梯度值，取平均梯度值进行更新
    # 随机梯度需要随机采样一个样本，计算其梯度值，进行更新
    # （其实这里不是正确的梯度上升，
    # 也不是随机梯度上升）
    for i in range(iter):
        for j in range(train_data.shape[0]):
            w_x=np.dot(weight,train_data[j])
            y_j=train_label[j]
            x_j=train_data[j]
            # 梯度上升
            weight+=lr*(x_j*y_j-(np.exp(w_x)*x_j)/(1+np.exp(w_x)))

    return weight


def test_model(test_data:list,test_lable,weight):
    '''
    计算准确率
    :param test_data: 测试集
    :param test_lable: 标签
    :param weight: 权重
    :return: 正确率
    '''
    # 增加一列1
    for i in range(len(test_data)):
        test_data[i].append(1)

    error_nums = 0

    for i in range(len(test_data)):
        if test_label[i]!=predict(weight,test_data[i]):
            error_nums+=1

    return 1-error_nums/len(test_data)


if __name__ == "__main__":

    # 加载数据
    test_data, test_label = load_data('./mnist_test/mnist_test.csv')
    train_data, train_label = load_data('./mnist_test/mnist_train.csv')

    start=time.time()
    # 开始训练，学习权重
    weights=logistic_regression(train_data,train_label)

    # 测试
    accuracy_rate=test_model(test_data,test_label,weights)
    print("the accuracy rate is:", accuracy_rate)

    end = time.time()
    print("time:",end-start)

