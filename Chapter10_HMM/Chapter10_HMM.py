# -*-coding:utf-8 -*-
# Author:Jaune
# Date:2020-5-27

'''
HMM是关于时序的概率模型，
主要有观测变量生成隐藏的状态序列，
然后再由状态变量生成观测值
主要步骤：1、概率计算模型，
给定模型参数值，计算观测序列出现的概率-->前向算法或者后向算法
2、给定观测序列，求模型参数值-->Baum-Welch算法
3、给定模型参数和观测序列，求隐藏的状态序列-->Viterbi算法
'''

import numpy as np
import time


def load_data(filename):
    '''
    加载数据
    :param filename:路径名
    :return: 文章内容
    '''

    article = []
    with open(filename, encoding='utf-8') as fr:
        for line in fr.readlines():
            # 去掉回车符\n
            line = line.strip()
            article.append(line)

    return article


def param_init(filename):
    '''
    估算相关参数，即相关矩阵，pi,A,B
    :param filename: 训练数据
    :return: 相关参数值
    '''

    # 定义标记符号
    # B：表示一个词的开头
    # M:表示一个词的中间字
    # E:表示一个词的结尾
    # S:表示一个字

    status_dict = {'B': 0, 'M': 1, 'E': 2, 'S': 3}

    # 状态向量PI
    PI = np.zeros(4)
    # 状态转移矩阵
    A = np.zeros((4, 4))
    # 观测概率矩阵B
    # 因为是中文分词，ord(汉字）可以找到对应的编码，
    # 这里用65536的空间保证所有汉字都能被存储
    B = np.zeros((4, 65536))

    with open(filename, encoding='utf-8') as fr:
        for line in fr.readlines():
            # 切分
            cur_line = line.strip().split()
            # 标记
            word_label = []
            for i in range(len(cur_line)):
                # 如果元素的长度为1，说明只有一个字，则标记为S
                if len(cur_line[i]) == 1:
                    label = 'S'
                else:
                    # 如果不为1，则开头标记为B，尾部为E
                    label = 'B' + 'M' * (len(cur_line[i]) - 2) + 'E'

                # 统计每行以某个词开头的概率，这次是在状态矩阵B[i][j]
                if i == 0:
                    PI[status_dict[label[0]]] += 1

                for j in range(len(label)):
                    # 观测概率？？？？
                    B[status_dict[label[j]]][ord(cur_line[i][j])] += 1

                word_label.extend(label)

            # 统计状态转移矩阵中的概率
            for i in range(1, len(word_label)):
                A[status_dict[word_label[i - 1]]][status_dict[word_label[i]]] += 1

        # 初始化状态矩阵
        sum_pi = np.sum(PI)
        for i in range(len(PI)):
            # 为了防止出现0,出现数据溢出的情况
            if PI[i] == 0:
                PI[i] = -3.14e+100
            else:
                # 取对数是因为概率值太小，出现数据溢出的情况
                PI[i] = np.log(PI[i] / sum_pi)

        # 初始化状态转移矩阵
        for i in range(len(A)):
            sum_a = np.sum(A[i])
            for j in range(len(A[i])):
                if A[i][j] == 0:
                    A[i][j] = -3.13e+100
                else:
                    A[i][j] = np.log(A[i][j] / sum_a)

        # 观测矩阵
        for i in range(len(B)):
            sum_b = np.sum(B[i])
            for j in range(len(B[i])):
                if B[i][j] == 0:
                    B[i][j] = -3.13e+100
                else:
                    B[i][j] = np.log(B[i][j] / sum_b)

    return PI, A, B


def divide_article(test_data, PI, A, B):
    '''
    分词
    :param test_data:测试集，要分词的文章
    :param PI: 初始状态概率向量
    :param A: 状态转移矩阵
    :param B: 观测矩阵
    :return:
    '''

    ret_artical = []
    for line in test_data:
        # P185-->算法10.5

        delta = [[0 for i in range(4)] for i in range(len(line))]
        # 初始化每种状态的δ
        for i in range(4):
            # 因为前面用的log,因此相加
            delta[0][i] = PI[i] + B[i][ord(line[0])]

        psi = [[0 for i in range(4)] for i in range(len(line))]

        for t in range(1, len(line)):
            for i in range(4):
                tmp_delta = [0] * 4
                for j in range(4):
                    tmp_delta[j] = delta[t - 1][j] + A[j][i]

                max_delta = max(tmp_delta)
                max_delta_indx = tmp_delta.index(max_delta)

                delta[t][i] = max_delta + B[i][ord(line[t])]
                psi[t][i] = max_delta_indx

        sequence = []
        # 获取最大概率的索引
        i_opt = delta[len(line) - 1].index(max(delta[len(line) - 1]))
        sequence.append(i_opt)
        # 开始回溯
        for t in range(len(line) - 1, 0, -1):
            i_opt = psi[t][i_opt]
            sequence.append(i_opt)
        sequence.reverse()

        # 开始分词
        cur_line = ""
        for i in range(len(line)):
            cur_line += line[i]

            if (sequence[i]==3 or sequence[i]==2) and i!=len(line)-1:
                cur_line +='|'

        ret_artical.append(cur_line)

    return ret_artical


if __name__ == "__main__":

    # 给定模型参数值
    PI, A, B = param_init("./HMM/HMMTrainSet.txt")
    # print(PI)

    # 加载要分词的数据
    data_set = load_data("./HMM/testArtical.txt")
    print(data_set[0])

    res = divide_article(data_set,PI,A,B)
    print(res[0])
