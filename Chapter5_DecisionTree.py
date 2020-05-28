# -*-coding:utf-8 -*-
# Author:Jaune
# Date:2020-5-27

'''
决策树可以用于分类与回归问题，本代码主要是分类问题，
主要思想，利用if-then规则的集合，呈树形结构，
步骤：特征选择-->根据信息增益或者信息增益比
生成决策树-->ID3:信息增益，C4.5:信息增益比,
CART:基尼指数(分类），回归，最小平方差
修剪决策树,此代码并未减枝
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

    # return np.array(data_arr), np.array(label_arr)
    return data_arr, label_arr


def empirical_entropy(train_label):
    '''
    计算经验熵H(D)
    :param train_label:训练集的标签
    :return: 经验熵
    '''
    # 统计哪些标签在训练集中，这样做的目的是确保标签一定在训练集中，
    # 假如某一类标签没有在训练集中出现，Ck=0,log0就没有意义，
    # 书上P61令0log0=0
    # train_label_set=[]
    # for i in train_label:
    #     if i not in train_label_set:
    #         train_label_set.append(i)
    #         continue

    train_label_set = set([label for label in train_label])
    h_d = 0
    for i in train_label_set:
        # size是计算array数组的元素个数，一维的数组的size=数组的长度
        # 计算每一个的个数
        counts = train_label[train_label == i].size
        prob = counts / train_label.size
        # 经验熵
        h_d += -1 * prob * np.log2(prob)

    return h_d


def conditional_entropy(train_data_column, train_label):
    '''
    计算条件熵H(Y|X)=sum  (pi*H(Y|X=xi)),i为某特征
    :param train_data_column:某一列特征
    :param train_label:标签
    :return: 条件熵
    '''

    h_d_a = 0
    # 计算feature的取值有哪些（其实是0，1）
    data_column_set = set([label for label in train_data_column])
    for i in data_column_set:
        # 特征取值为i的样本数
        count_i = train_data_column[train_data_column == i].size
        h_d_a += count_i / train_data_column.size \
                 * empirical_entropy(train_label[train_data_column == i])

    return h_d_a


def best_feature(train_data, train_label):
    '''
    计算信息增益最大的特征
    :param train_data: 训练集
    :param train_label: 训练集标签
    :return: 信息增益最大的特征及最大信息增益值
    '''
    # 一共784维
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    feature_nums = train_data.shape[1]
    max_gda = -1
    max_gda_feature = -1
    h_d = empirical_entropy(train_label)
    for feature in range(feature_nums):
        # 某一维特征，
        train_data_feature = train_data[:, feature]
        g_d_a = h_d - conditional_entropy(train_data_feature, train_label)
        if g_d_a > max_gda:
            max_gda = g_d_a
            max_gda_feature = feature

    # 返回具有信息增益最大的特征及最大信息增益值
    return max_gda_feature, max_gda


def divide_data(data_arr, label_arr, feature, k):
    '''
    根据具有最大信息增益的那个特征进行切分数据集
    :param data_arr: 要切分的数据集
    :param label_arr: 对应的标签
    :param feature: 具有最大信息增益的特征
    :param k: 指定取值
    :return: 新的数据集及标签
    '''
    ret_data_arr = []
    ret_label_arr = []
    for i in range(len(data_arr)):
        # 如果样本的feature取值==k,则分在一边
        # 比如feature=3,k=0表示此时以第三维的特征划分，
        # 同时去掉这一维，返回值新的数据集和标签集
        if data_arr[i][feature] == k:
            ret_data_arr.append(list(data_arr[i][0:feature]) + list(data_arr[i][feature + 1:]))
            ret_label_arr.append(label_arr[i])

    return ret_data_arr, ret_label_arr


def max_classes(label_arr):
    '''
    统计标签集中个数最多的标签
    :param label_arr: 标签集
    :return: 个数最多的标签
    '''
    # 用字典存放个数，比如3：5表示标签为3的出现了5次
    # class_dict = {5: 0, 1: 0, 2: 0, 3: 0, 4: 0,
    #               0: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    class_dict = {}
    for i in range(len(label_arr)):
        if label_arr[i] in class_dict.keys():
            class_dict[label_arr[i]] += 1
        else:
            class_dict[label_arr[i]] = 1
    # 对字典元素进行排序，按次数从大到小排
    # 如果这里存在某2个标签的个数相等，则根据键的出现顺序排序
    class_sort = sorted(class_dict.items(), key=lambda x: x[1], reverse=True)
    # 因此这里取得节点类型有点不合理
    return class_sort[0][0]


def create_tree(*data_set):
    '''
    创建决策树
    :param data_set: 因为要递归调用，传入的是两个值，
    因此用元组的形式((train_data,train_label))，方便后续处理
    :return:新的子节点
    '''
    # 这里认为设置
    epsilon = 0.1
    # 这里取到数据集
    train_data = data_set[0][0]
    # aa= train_data[0]
    # bb = len(train_data[0])
    # 取到标签集
    train_label = data_set[0][1]
    #
    print("create a node ", len(train_data[0]), len(train_label))

    # 将标签转成集合，去重，
    class_dict = {i for i in train_label}
    # 如果标签个数等于1，表示数据集只有一个类，则不需要在分
    if len(class_dict) == 1:
        return train_label[0]
    # 如果没有特征了，返回实例中最多的类作为该节点的类
    if len(train_data[0]) == 0:
        return max_classes(train_label)

    max_gda_feature, max_gda = best_feature(train_data, train_label)
    if epsilon > max_gda:
        return max_classes(train_label)

    # 用字典存树
    tree_dict = {max_gda_feature: {}}
    tree_dict[max_gda_feature][0] = create_tree(
        divide_data(train_data, train_label, max_gda_feature, 0))

    tree_dict[max_gda_feature][1] = create_tree(
        divide_data(train_data, train_label, max_gda_feature, 1))

    return tree_dict


def predict(features, tree):
    '''
    预测标签
    :param features:一个数据
    :param label: 标签
    :param tree: 决策树
    :return: 预测结果
    '''

    while True:
        (key, value), = tree.items()
        # 进行判断，一层一层往下走，key为划分特征（也就是哪一维），
        # 特征取值为0，或者1，data_val=0,说明往0的分支走，反之往1走
        if type(tree[key]).__name__ == 'dict':
            data_val = features[key]
            del features[key]
            tree = value[data_val]
            if type(tree).__name__ == 'int':
                return tree
        else:
            return value


def test_model(test_data, test_label, tree):
    '''
    测试准确率
    :param test_data:测试数据集
    :param test_label: 标签
    :param tree: 决策树树
    :return: 准确率
    '''
    error_counts = 0
    for i in range(len(test_data)):
        if test_label[i] != predict(test_data[i], tree):
            error_counts += 1

    return 1 - error_counts / len(test_data)


if __name__ == "__main__":


    # 记载数据
    test_data, test_label = load_data("./mnist_test/mnist_test.csv")
    train_data, train_label = load_data("./mnist_test/mnist_train.csv")

    # 生成树
    start = time.time()
    # decision_tree = create_tree((test_data, test_label))
    decision_tree = create_tree((train_data, train_label))

    # 计算准确率
    # accuracy_rate = test_model(test_data[:350], test_label[:350],decision_tree)
    accuracy_rate = test_model(test_data, test_label,decision_tree)

    print("the accuracy rate is:",accuracy_rate)

    end = time.time()
    print(" take time:",end-start)


    # test_data = np.array(test_data)
    # test_label = np.array(test_label)

    # A = np.array([1, 2, 2, 5, 7, 5, 6, 7, 8, 9, 0, 0])
    # # res = max_classes(A)
    # print(type(test_data),len(test_data))
    # b = {item for item in A}
    # print(len(b), b)
