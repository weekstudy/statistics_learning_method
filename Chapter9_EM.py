# -*-coding:utf-8 -*-
# Author:Jaune
# Date:2020-5-30

'''
EM算法：用于对含有隐变量的概率模型进行参数估计

'''

import numpy as np
import random
import math
import time


def load_data(mu_0, sigma_0, mu_1, sigma_1, alpha_0, alpha_1):
    '''
    构造数据集，这里通过高斯混合模型来构造数据
    :param mu_0: 均值
    :param sigma_0:标准差
    :param mu_1:
    :param sigma_1:
    :param alpha_0:混合高斯的系数
    :param alpha_1:
    :return:高斯混合模型的数据
    '''

    # 1000个数据
    length = 1000
    data_0 = np.random.normal(mu_0, sigma_0, int(length * alpha_0))
    data_1 = np.random.normal(mu_1, sigma_1, int(length * alpha_1))

    data_set = []
    # 放在一起，并捣乱
    data_set.extend(data_0)
    data_set.extend(data_1)
    random.shuffle(data_set)

    return data_set


def gauss_model(data_arr, mu, sigma):
    '''
    根据混合高斯密度函数计算概率值
    :param data_arr: 观测数据
    :param mu: 均值
    :param sigma: 标准差
    :return:观测值的概率
    '''
    # P162-->式9.25
    res = (1 / (math.sqrt(2 * math.pi) * sigma)) * \
          np.exp(-1 * (data_arr - mu) * (data_arr - mu) / (2 * sigma ** 2))

    return res


def expectation(data_arr, alpha_0, mu_0, sigma_0, alpha_1, mu_1, sigma_1):
    '''
    EM算法的中第一步求期望值
    :param data_arr: 观测数据y
    :param alpha_0:
    :param mu_0:
    :param sigma_0:
    :param alpha_1:
    :param mu_1:
    :param sigma_1:
    :return: 观测数据来自第k个高斯模型的概率
    '''

    # P164
    gamma_0 = alpha_0 * gauss_model(data_arr, mu_0, sigma_0)
    gamma_1 = alpha_1 * gauss_model(data_arr, mu_1, sigma_1)

    normalization = gamma_0 + gamma_1
    gamma_0 = gamma_0 / normalization
    gamma_1 = gamma_1 / normalization

    return gamma_0, gamma_1


def max_expectation(mu_0, mu_1, gamma_0, gamma_1, data_arr):
    # p165-->9.2算法
    mu_0_new = np.dot(gamma_0, data_arr) / np.sum(gamma_0)
    mu_1_new = np.dot(gamma_1, data_arr) / np.sum(gamma_1)

    sigma_0_new = math.sqrt(np.dot(gamma_0, (data_arr - mu_0) ** 2) / np.sum(gamma_0))
    sigma_1_new = math.sqrt(np.dot(gamma_1, (data_arr - mu_1) ** 2) / np.sum(gamma_1))

    alpha_0_new = np.sum(gamma_0) / len(gamma_0)
    alpha_1_new = np.sum(gamma_1) / len(gamma_1)

    return mu_0_new, mu_1_new, sigma_0_new, sigma_1_new, alpha_0_new, alpha_1_new


def train_model(data_set, iter=500):
    '''
    进行迭代
    :param data_set:
    :param iter:
    :return:
    '''

    # 初始化参数
    data_arr = np.array(data_set)
    alpha_0 = 0.5
    mu_0 = 0
    sigma_0 = 1
    alpha_1 = 0.5
    mu_1 = 1
    sigma_1 = 1

    step = 0
    while (step < iter):
        step += 1
        gamma_0, gamma_1 = expectation(data_arr, alpha_0, mu_0, sigma_0,
                                       alpha_1, mu_1, sigma_1)

        mu_0, mu_1, sigma_0, sigma_1, alpha_0, alpha_1 = max_expectation(
            mu_0, mu_1, gamma_0, gamma_1, data_arr)

    return alpha_0, mu_0, sigma_0, alpha_1, mu_1, sigma_1


if __name__ == "__main__":

    # 真实分布
    alpha0 = 0.3
    mu0 = -2
    sigma0 = 0.5
    alpha1 = 0.7
    mu1 = 0.5
    sigma1 = 1

    data_set = load_data(mu0, sigma0, mu1, sigma1, alpha0, alpha1)

    # 真实参数值
    print('alpha0:%.1f, mu0:%.1f, sigmod0:%.1f, alpha1:%.1f, mu1:%.1f, sigmod1:%.1f' % (
        alpha0, mu0, sigma0, alpha1, mu1, sigma1))

    # 估计参数值
    alpha0, mu0, sigma0, alpha1, mu1, sigma1 = train_model(data_set)

    print('alpha0:%.1f, mu0:%.1f, sigmod0:%.1f, alpha1:%.1f, mu1:%.1f, sigmod1:%.1f' % (
        alpha0, mu0, sigma0, alpha1, mu1, sigma1))
