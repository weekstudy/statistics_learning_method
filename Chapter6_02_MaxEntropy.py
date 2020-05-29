# -*-coding:utf-8 -*-
# Author:Jaune
# Date:2020-5-29

'''
最大熵模型原理：所有模型中，熵最大的模型是最好的模型，
也就是满足约束条件的模型集合中选熵最大的模型
由于约束函数数量和样本数目有关系，
导致迭代过程计算量巨大，本测试代码用笔记本跑了40min,就没跑了。
步骤1.计算联合分布p(x,y)f(x,y)的期望值
2.计算条件分布p(x)p(y|x)f(x,y)的期望值
3.开始迭代
'''

from collections import defaultdict
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
            if int(cur_line[0]) == 0:
                label_arr.append(1)
            else:
                label_arr.append(0)

    return data_arr, label_arr


class MaxEntropy(object):
    '''
    最大熵类
    '''

    def __init__(self, train_data, train_label, test_data, test_label):
        '''
        初始化数据
        :param train_data: 训练集
        :param train_label: 训练集标签
        :param test_data: 测试集
        :param test_label: 测试集标签
        '''

        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        # 特征数
        self.features = len(train_data[0])
        # 训练样本数
        self.N = len(train_data)
        self.n = 0
        # 改进的迭代尺度算法中6.34的f^#(x,y)=M,
        # 表示所有特征在(x,y)出现的次数,这里有点绕
        self.M = 10000

        # 每维特征取值相同的次数，
        # 比如需要统计f1（1，0）表示第一维取值为1，标签为0的次数
        self.xi_y_pairs = self.func_xi_y()
        self.weights = [0] * self.n
        self.xy2id_dict, self.id2xy_dict = self.create_search_dict()
        # 计算联合分布的期望值
        self.Ep_xy = self.calc_expected()

    def calc_expected(self):
        '''
        计算特征函数f(x,y)关于联合经验分布p(x，y)的期望值
        :return:
        '''

        ep_xy = [0] * self.n
        for i in range(self.features):
            for (x, y) in self.xi_y_pairs[i]:
                id = self.xy2id_dict[i][(x, y)]
                ep_xy[id] = self.xi_y_pairs[i][(x, y)] / self.N

        return ep_xy

    def func_xi_y(self):
        '''
        计算（x_i,y)在训练集的次数
        :return:
        '''
        f_i_xi_y = [defaultdict(int) for i in range(self.features)]

        for i in range(len(self.train_data)):
            for j in range(self.features):
                # 将配对的加入字典，计数,列表里面放的是字典，
                # j表示第j个字典，后面才是键
                f_i_xi_y[j][(self.train_data[i][j], self.train_label[i])] += 1
        # 统计整个有多少对
        for i in f_i_xi_y:
            self.n += len(i)

        return f_i_xi_y

    def create_search_dict(self):
        ''''
        创建查询字典
        xy2id_dict:通过（x,y)对应的id,
        id2xy_dict:通过id对应的（x,y)

        '''
        # 这里的x是指一维的特征，这里有784维，因此有784个字典
        # 比如第3维特征值组合有（0，0），（0，1），(1,0),（1，1）
        # 但是并不能通过（1，1）找到id是第三维，有可能第四维也是（1，1）
        xy2id_dict = [{} for i in range(self.features)]

        id2xy_dict = {}

        # 建立索引
        index = 0
        for i in range(self.features):
            for (x, y) in self.xi_y_pairs[i]:
                xy2id_dict[i][(x, y)] = index
                id2xy_dict[index] = (x, y)

                index += 1

        return xy2id_dict, id2xy_dict

    def calc_prob_y_x(self, x, y):
        '''
        计算模型的p(y|x)
        :param x: 一个样本，784维
        :param y: 标签
        :return: Pw(y|x)
        '''

        # 分子
        numerator = 0
        # 分母
        normalization = 0
        for i in range(self.features):
            if (x[i], y) in self.xy2id_dict[i]:
                index = self.xy2id_dict[i][(x[i], y)]
                numerator += self.weights[index]

            if (x[i], 1 - y) in self.xy2id_dict[i]:
                index = self.xy2id_dict[i][(x[i], 1 - y)]
                normalization += self.weights[index]
        # 分子
        numerator = np.exp(numerator)
        # 分母
        normalization = np.exp(normalization) + numerator

        # 返回p(y|x)值
        return numerator / normalization

    def train_model(self, iter=500):
        '''
        训练模型
        :param iter: 迭代次数
        :return:
        '''
        for i in range(iter):
            # 计算模型的p(y|x)p(x)f_i(x_i,y)期望值
            exp_y_x = self.calc_cond_expected()

            # 采用改进的迭代尺度法
            sigma_list = [0] * self.n
            for j in range(self.n):
                # 这里假设了为常数,6.34的公式
                sigma_list[j] = (1 / self.M) * np.log(self.Ep_xy[j] / exp_y_x[j])

            # 更新权重
            self.weights = [self.weights[i] + sigma_list[i] for i in range(self.n)]

    def calc_cond_expected(self):
        '''
        计算模型的p(y|x)p(x)f_i(x_i,y)期望值
        # 书中的x应该是指一维特征，而本例中有784维，
        # 因此特征函数f(x,y)就有784个
        :return:
        '''
        cond_expected = [0] * self.n
        for i in range(self.N):
            p_y_x = [0] * 2
            p_y_x[0] = self.calc_prob_y_x(self.train_data[i], 0)
            p_y_x[1] = self.calc_prob_y_x(self.train_data[i], 1)

            for j in range(self.features):
                for y in range(2):
                    if (self.train_data[i][j], y) in self.xi_y_pairs[j]:
                        id = self.xy2id_dict[j][(self.train_data[i][j], y)]
                        cond_expected[id] += (1 / self.N) * p_y_x[y]

        return cond_expected

    def predict(self, x):
        '''
        预测标签
        :param x:要预测的样本
        :return: 预测结果
        '''
        result = [0] * 2

        for i in range(2):
            result[i] = self.calc_prob_y_x(x, i)

        # 返回值
        return result.index(max(result))

    def test_model(self):
        '''
        用测试集进行测试
        :return: 准确率
        '''
        error_nums = 0
        for i in range(len(self.test_data)):
            result = self.predict(self.test_data[i])

            if result != self.test_label[i]:
                error_nums += 1

        return 1 - error_nums / len(self.test_data)


if __name__ == "__main__":
    # 加载数据
    test_data, test_label = load_data('./mnist_test/mnist_test.csv')
    train_data, train_label = load_data('./mnist_test/mnist_train.csv')

    # 因原数据集太大,因此就只采用100个样本训练,测试采用10个数据
    max_ent = MaxEntropy(train_data[:10], train_label[:10], test_data[:10], test_label[:10])

    start = time.time()
    # 开始训练
    max_ent.train_model()

    # 准确率
    accuracy = max_ent.test_model()
    print(accuracy)

    end = time.time()
    print("time:", end - start)
