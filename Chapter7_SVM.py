# -*-coding:utf-8 -*-
# Author:Jaune
# Date:2020-5-27

'''
SVM主要思想:二分类模型,在特征空间上进行划分,
使样本到超平面的间隔最大,
'''

import numpy as np
import time
import math
import random


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
            # 为了简化，就做二分类，将标签为0的标为1，其他都为-1，
            # 其实也可以做多分类
            if int(cur_line[0]) == 0:
                label_arr.append(1)
            else:
                label_arr.append(-1)

    # return np.array(data_arr), np.array(label_arr)
    return data_arr, label_arr


class Svm(object):
    '''
    SVM类
    '''

    def __init__(self, train_data, train_label, sigma=10,
                 penalty_coef=200, slack_var=0.001):
        '''
        svm相关参数设置
        :param train_data:训练集
        :param train_label: 标签
        :param sigma: 高斯核中标准差
        :param penalty_coef: 惩罚系数
        :param slack_var: 松弛变量
        '''

        # 为了方便计算,转为矩阵
        self.data_mat = np.mat(train_data)
        self.label_mat = np.mat(train_label).T

        self.m, self.n = self.data_mat.shape

        self.sigma = sigma
        self.penalty_coef = penalty_coef
        self.slack_var = slack_var
        # 初始化核矩阵
        self.kernel = self.cal_kernel()
        self.bias = 0
        # 对偶算法中的系数
        self.alpha = [0] * self.m
        # SMO算法中的预测值与真实值之差
        self.error_val = [0 * self.data_mat[i, 0] for i in range(self.m)]
        self.support_vec_index = []

    def cal_kernel(self):
        '''
        计算核函数
        :return:高斯核矩阵
        '''

        kernel = [[0 for i in range(self.m)] for j in range(self.m)]
        for i in range(self.m):
            if i % 100 == 0:
                print('construct the kernel:', i, self.m)
            # 7.90式的x
            x = self.data_mat[i, :]
            for j in range(self.m):
                # 7.90式的z
                z = self.data_mat[j, :]
                res = (x - z) * (x - z).T
                res = np.exp(-1 * res / (2 * self.sigma ** 2))

                kernel[i][j] = res
                kernel[j][i] = res

        return kernel

    def is_kkt(self, i):
        '''
        验证alpha_i是否满足KKT条件
        :param i: 下标i
        :return:true :满足false:不满足
        '''

        pred_y_i = self.pred_y(i)
        y_i = self.label_mat[i]
        # 7.111
        if (math.fabs(self.alpha[i]) < self.slack_var) and y_i * pred_y_i >= 1:
            return True
        # 7.113
        elif (math.fabs(self.alpha[i] - self.penalty_coef) < self.slack_var) \
                and y_i * pred_y_i <= 1:
            return True
        # 7.112
        elif self.alpha[i] > -self.slack_var and self.alpha[i] < (self.penalty_coef + self.slack_var) \
                and math.fabs(y_i * pred_y_i - 1) < self.slack_var:
            return True

        return False

    def pred_y(self, i):
        '''
        计算预测值pred_y_i
        :param i: 下标
        :return: 预测值pred_y_i
        '''
        # 只计算alpha不为0的样本

        pred_y_i = 0
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        # 式7.104
        for j in index:
            pred_y_i += self.alpha[j] * self.label_mat[j] * self.kernel[j][i]

        # 加上偏置
        pred_y_i += self.bias

        return pred_y_i

    def calc_error_val(self, i):
        '''
        计算预测值与真实值的差值,式7.105
        :param i: 样本下标
        :return:
        '''
        pred_y_i = self.pred_y(i)

        return pred_y_i - self.label_mat[i]

    def get_alpha_second(self, error1, i):
        '''
        选择第二个迭代变量
        :param error1:第一个变量取alpha_i后的预测值与真实值之间的差值
        :param i:下标
        :return: error2,i第二个迭代变量的下标
        '''

        error2 = 0
        maxE1_E2 = -1
        max_index = -1

        # 取得非零的E_i和下标
        nozeroE = [i for i, E_i in enumerate(self.error_val) if E_i != 0]
        for j in nozeroE:
            # P129-->第二个变量的选择
            error2_tmp = self.calc_error_val(j)
            if math.fabs(error1 - error2_tmp) > maxE1_E2:
                maxE1_E2 = math.fabs(error1 - error2_tmp)
                error2 = error2_tmp
                max_index = j

        # 如果列表中没有非零元素,则随机选一个
        if max_index == -1:
            max_index = i
            while max_index == i:
                max_index = int(random.uniform(0, self.m))

            error2 = self.calc_error_val(max_index)

        return error2, max_index

    def train_model(self, iters=100):
        '''
        开始训练
        :param iters: 迭代次数
        :return:
        '''
        # 迭代次数,超过设置次数还未收敛则强制停止
        iter_step = 0
        # param_changed:单次迭代中,有参数改变就+1
        parm_changed = 1

        while (iter_step < iters) and parm_changed > 0:
            # 打印当前迭代次数
            print("iter:%d/%d" % (iter_step, iters))
            iter_step += 1
            # 将标志位重置0
            parm_changed = 0

            for i in range(self.m):
                # 不满足kkT条件,则作为SMO的第一个变量
                if self.is_kkt(i) == False:

                    error1 = self.calc_error_val(i)
                    error2, j = self.get_alpha_second(error1, i)

                    # 参考7.4.1两个变量二次规划的求解方法
                    y1 = self.label_mat[i]
                    y2 = self.label_mat[j]

                    alpha_old1 = self.alpha[i]
                    alpha_old2 = self.alpha[j]

                    if y1 != y2:
                        L = max(0, alpha_old2 - alpha_old1)
                        H = min(self.penalty_coef, self.penalty_coef + alpha_old2 - alpha_old1)
                    else:
                        L = max(0, alpha_old2 + alpha_old1 - self.penalty_coef)
                        H = min(self.penalty_coef, alpha_old2 + alpha_old1)
                    #  如果相等,则说明无法优化当前变量,继续下一轮
                    if L == H: continue

                    # 计算alpha的新值
                    k11 = self.kernel[i][i]
                    k12 = self.kernel[i][j]
                    k21 = self.kernel[j][i]
                    k22 = self.kernel[j][j]
                    # p127-->式7.106
                    alpha_new2 = alpha_old2 + y2 * (error1 - error2) / (k11 + k22 - 2 * k12)
                    # 剪切
                    if alpha_new2 < L:
                        alpha_new2 = L
                    elif alpha_new2 > H:
                        alpha_new2 = H
                    # 更新alpha_1,式7.109
                    alpha_new1 = alpha_old1 + y1 * y2 * (alpha_old2 - alpha_new2)

                    # 计算b_new1,b_new2,,P129-->式7.115
                    b_new1 = -1 * error1 - y1 * k11*(alpha_new1 - alpha_old1) \
                             - y2 * k21 * (alpha_new2 - alpha_old2) + self.bias
                    b_new2 = -1 * error2 - y1 * k12*(alpha_new1 - alpha_old1) \
                             - y2 * k22*(alpha_new2 - alpha_old2) + self.bias

                    # 进行判断新bias
                    if 0 < alpha_new1 < self.penalty_coef:
                        b_new = b_new1
                    elif 0 < alpha_new2 < self.penalty_coef:
                        b_new = b_new2
                    else:
                        b_new = (b_new1 + b_new2) / 2

                    # 进行更新alpha,bias
                    self.alpha[i] = alpha_new1
                    self.alpha[j] = alpha_new2
                    self.bias = b_new
                    self.error_val[i] = self.calc_error_val(i)
                    self.error_val[j] = self.calc_error_val(j)

                    # 如果alpha_2的值改变太小,则重新选择,
                    if math.fabs(alpha_new2 - alpha_old2) >= 0.00001:
                        parm_changed += 1

                # 打印迭代轮数
                print("iter:%d i:%d, pairs changed %d" % (iter_step, i, parm_changed))

        # 记录支持向量
        for i in range(self.m):
            if self.alpha[i] > 0:
                self.support_vec_index.append(i)

    def kernel_func(self,x1,x2):
        '''
        核函数
        :param x1:
        :param x2:
        :return:
        '''
        result=(x1-x2)*(x1-x2).T
        result=np.exp(-1*result/(2*self.sigma**2))

        return result

    def predict(self,x):
        '''

        :param x:样本
        :return: 预测结果
        '''
        res=0
        for i in self.support_vec_index:
            temp=self.kernel_func(self.data_mat[i,:],np.mat(x))
            # 预测P124 -->式7.94
            res+=self.alpha[i]*self.label_mat[i]*temp

        res+=self.bias
        return np.sign(res)

    def test_model(self,test_data,test_lable):
        '''
        测试模型
        :param test_data: 测试集
        :param test_lable: 标签
        :return: 正确率
        '''

        error_nums=0
        for i in range(len(test_data)):
            # 打印进度
            print("test:%d/%d" % (i,len(test_data)))
            res=self.predict(test_data[i])
            if res!=test_label[i]:
                error_nums+=1

        return 1-error_nums/len(test_data)



if __name__ == "__main__":
    # 加载数据
    test_data, test_label = load_data('./mnist_test/mnist_test.csv')
    train_data, train_label = load_data('./mnist_test/mnist_train.csv')

    # 初始化svm类
    svm = Svm(train_data[:10], train_label[:10],10,200,0.001)

    start=time.time()

    # 开始训练
    svm.train_model()
    # 开始测试
    accuracy=svm.test_model(test_data[:100],test_label[:100])

    end=time.time()
    print("the accuracy rate is:",accuracy)
    print("time:",end-start)
