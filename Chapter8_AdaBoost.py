# -*-coding:utf-8 -*-
# Author:Jaune
# Date:2020-5-27

'''
AdaBoost:通过组合不同权重的弱分类器,构成一个强分类器
构建弱分类器：如果当前样本分错，则下一轮增加分错样本的权值，
如果当前样本分对，则下一轮降低样本的权值，以构建分类器
权重：增加弱分类器误差小的权值，在表决过程中起重要作用，
反之，减小作用
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
            data_arr.append([int(int(num) / 128) for num in cur_line[1:]])
            # 为了简化，就做二分类，将标签为0的标为1，其他都为-1，
            # 其实也可以做多分类
            if int(cur_line[0]) == 0:
                label_arr.append(1)
            else:
                label_arr.append(-1)

    # return np.array(data_arr), np.array(label_arr)
    return data_arr, label_arr


def classify(train_data, train_label, feature_n, div, rule, d):
    '''
    计算分类错误率
    :param train_data:训练集
    :param train_label: 标签
    :param feature_n: 第n维特征
    :param div: 划分点阈值
    :param rule: 正反例标签
    :param d: 权值分布
    :return: 分类错误率
    '''

    error_nums = 0
    feature = train_data[:, feature_n]
    y = train_label
    predict = []

    if rule == "low_is_one":
        L = 1
        H = -1
    else:
        L = -1;H = 1

    for i in range(train_data.shape[0]):
        if feature[i] < div:
            predict.append(L)
            if y[i] != L:
                error_nums += d[i]
        elif feature[i]>=div:
            predict.append(H)
            if y[i]!=H:
                error_nums+=d[i]

    return np.array(predict), error_nums


def create_single_tree(train_data, train_label, d):
    '''
    创建单层提升树
    :param train_data: 训练集
    :param train_label: 标签
    :param d: 权值分布
    :return:
    '''

    m, n = np.shape(train_data)
    # 单层树
    single_tree = {}
    # 初始化误差率最多是100%，因此为1
    single_tree['e'] = 1

    for i in range(n):
        # 因为特征取值经过二值化，只能为0，1，因此阈值为-0.5，0.5，1，5
        for div in [-0.5, 0.5, 1.5]:
            # 在进行划分时，有2中情况：
            # 小于某值的为1，大于某值的为-1，low_is_one
            # 小于某值的为-1，大于某值的为1,high_is_one
            for rule in ['low_is_one', 'high_is_one']:
                # 以i特征划分时的预测结果和分类错误率
                classify_res, e = classify(train_data, train_label, i, div, rule, d)
                # 如果错误率小，则保存
                if e < single_tree['e']:
                    single_tree['e'] = e
                    # 保存划分特征，划分规则，结果，特征索引
                    single_tree['div'] = div
                    single_tree['rule'] = rule
                    single_tree['classify_res'] = classify_res
                    single_tree['feature'] = i

    return single_tree


def create_adaboosting_tree(train_data, train_label, tree_nums=50):
    '''
    创建提升树
    :param train_data:训练集
    :param train_lable: 标签
    :param tree_nums: 树的层数
    :return:
    '''

    train_data_arr = np.array(train_data)
    train_label_arr = np.array(train_label)
    # 每增加一层，当前预测结果列表
    last_pred = [0] * len(train_data_arr)
    m, n = np.shape(train_data_arr)
    # 初始化权重
    d = [1 / m] * m

    adaboost_tree = []
    # P138-->8.1.2
    for i in range(tree_nums):
        cur_tree = create_single_tree(train_data_arr, train_label_arr, d)
        alpha = 1/2*np.log((1-cur_tree['e'])/cur_tree['e'])
        classify_res=cur_tree['classify_res']
        # 更新权值分布
        d=np.multiply(d,np.exp(-1*alpha*np.multiply(train_label_arr,classify_res)))/sum(d)
        cur_tree['alpha']=alpha
        adaboost_tree.append(cur_tree)

        # 以下代码用来测试的，可以删掉
        last_pred+=alpha*classify_res
        error=sum([1 for i in range(len(train_data)) if np.sign(last_pred[i])!=train_label_arr[i]])
        last_error=error/len(train_data)
        # 如果误差为0，则返回树
        if last_error==0:
            return adaboost_tree
        print("iter:%d:%d,single_tree error:%.4f,last error:%.4f" % (i,tree_nums,cur_tree['e'],last_error))

        return adaboost_tree


def predict(x,div,rule,feature):
    '''
    输出一层的预测结果
    :param x: 预测样本
    :param div: 划分特征
    :param rule: 划分规则
    :param feature:进行操作的特征
    :return:
    '''

    if rule == 'low_is_one':
        L=1;H=-1
    else:L=-1;H=1

    if x[feature]<div:
        return L
    else:
        return H


def test_model(test_data,test_label,tree):
    '''
    测试
    :param test_data: 测试数据集
    :param test_label: 标签
    :param tree: 提升树
    :return: 准确率
    '''

    error_nums=0
    for i in range(len(test_data)):
        res =0
        for cur_tree in tree:
            div=cur_tree['div']
            rule=cur_tree['rule']
            feature=cur_tree['feature']
            alpha=cur_tree['alpha']

            res+=alpha*predict(test_data[i],div,rule,feature)
        if np.sign(res)!=test_label[i]:
            error_nums+=1

    return 1-error_nums/len(test_data)




if __name__ == "__main__":
    # 加载数据
    test_data, test_label = load_data('./mnist_test/mnist_test.csv')
    train_data, train_label = load_data('./mnist_test/mnist_train.csv')

    start=time.time()

    # 为了节约时间，选用100组数据创建提升树
    adaboosting_tree=create_adaboosting_tree(train_data[:100],train_label[:100],40)

    # 准确率
    accuracy=test_model(test_data[:100],test_label[:100],adaboosting_tree)
    print("the accuracy rate is:",accuracy)

    end = time.time()
    print("time:",end-start)
