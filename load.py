#数据处理及分配
#读取train.csv和test.csv，划分样本的特征和标签，数据编码，数据标准化，边缘端数据划分

import numpy as np
import pandas as pd
from sklearn import  preprocessing
from sklearn.model_selection import StratifiedKFold
import warnings


def getdata(class_num):  #  边缘端数据划分

    features_train = pd.read_csv('train.csv')
    #testData = pd.read_csv('test.csv')

    all_labels_train = np.array(features_train['actual'])  #  标签   也就是要预测的温度的真实值  （划分样本的特征和标签）
    features_train = features_train.drop('actual', axis = 1)  #  在特征中去掉标签    （划分样本的特征和标签）
    features_train = pd.get_dummies(features_train)  #  独热编码  将week中的Fri、Sun等编码而不是String格式    （数据编码）
    all_features_train = preprocessing.StandardScaler().fit_transform(features_train)  #  数据标准化
    # all_labels_train和all_features_train就是要用的


    # 返回所有特征和标签
    return all_features_train


#测试    index_class = getdata(20)
