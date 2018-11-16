# -*- coding:utf-8 -*-
"""
@author:xuyi
@time:2018/9/4  15:55
"""
from sklearn.preprocessing import Imputer, scale
from sklearn.datasets import load_iris
import numpy as  np
from scipy.interpolate import lagrange
from myapp.dataDeal.NewTonMethod import NewtonInterpolation
import math

#自定义列向量插值函数
#s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
def ployinterp_column(s, n, method="lagrange", k=6):
    y = s[list(range(n-k, n)) + list(range(n+1, n+1+k))] #取数
    # y = y[y.notnull()] #剔除空值
    index = []
    for i, value in enumerate(y):
        index.append(i+1)
    if method == "lagrange":
        return lagrange(index, y)(n) #插值并返回插值结果
    elif method == "NewtonMethod":
        nt = NewtonInterpolation()
        return nt.calculate_data(n,index,y)[0]

def fix_data(data, method="mean", nList=None):
    """
    数据进行修复处理
    :param data: 原始数据
    :param n: 不正常数据的位置
    :param method: 进行数据修复的方法，有均值，取中值，拉格朗日插值，牛顿插值法
    :return: 进行修复过的数据
    """


    if isinstance(data, list):
        data = np.asarray(data, dtype=float)
    # 制作一份副本
    X_t = data.copy()
    if len(data.shape) == 1:
        if nList is None:
            nList = []
            for k, v in enumerate(data):
                if v == "" or v == None or v != v or v == 0:
                    nList .append(k)
    # 修补数据
    if method == "mean" or method == "median" or method == "most_frequent":
        # 创建一个imputer 对象，采用平均值策略
        imputer = Imputer(missing_values=0, strategy= method)
        X_imputer = imputer.fit_transform(X_t)
        return X_imputer
    elif method == 'lagrange' or method == 'NewtonMethod':
        if len(X_t.shape) == 2:
            # 在第二行制作一些丢失值
            X_t[10, :] = np.repeat("NaN", data.shape[1])  # 维数的第几列
            print(data.shape[1])
            for i in range(4):
                 for j, value in enumerate (X_t[:, i]):
                    if value != value:
                        X_t[j, i] = ployinterp_column(X_t[:, i], j ,method=method)
        elif len(X_t.shape) == 1:
            for k in nList:
                X_t[k] = ployinterp_column(X_t,k,method= method)
        return X_t

def RandomSample(data,no_records):
    """随机取样"""
    print(isinstance(data,np.ndarray))
    if isinstance(data,list):
        x = np.asarray(data)
    elif isinstance(data,np.ndarray):
        x = data
    x_sample_idx = np.random.choice(range(x.shape[0]), no_records)
    return x_sample_idx

def Standard(data):
    # 标准化
    return scale(data, with_mean=True, with_std=True)

def variance(data, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    """求方差"""
    return np.var(data, axis, dtype, out, ddof, keepdims)
def getMean(a, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    """求均值"""
    return np.mean(a, axis, dtype, out, keepdims)
def getMedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """求中位数"""
    return np.median(a, axis, out, overwrite_input, keepdims)

def getStandardDeviation(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    """标准差"""
    return np.std(a, axis, dtype, out, ddof, keepdims)

def coefficientOfDispersion(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue, overwrite_input=False):
    """变异系数"""

    std = getStandardDeviation(a)
    mean = getMean(a, axis)
    print(mean)
    return std / mean
def create_dataset(source_data, look_back=5):
    """
    # 创建数据集，每五个为一组，每组的第二个数为下一组的第一个数， 前面那四个是输入，最后一个数是输出 (只适合一维数组)
    :param data: 待处理数据
    :param look_back: 几个为一组
    :return: 处理之后的数据

    """
    X = []
    for i, v in enumerate(source_data[look_back:], look_back):  # 遍历序列，可以有下标，下标从look_back 开始
        X += [source_data[i - look_back:i + 1]]
        dataset = np.asarray(X)
        train_data = dataset[:, 0:look_back]
        label_data = dataset[:, -1]
    return dataset ,train_data, label_data

def min_max_normalize(source_data):
    """
        数据归一化
        :param dataset: 待处理数据
        :return: 归一化之后的数据
        """
    min_v = np.min(source_data)
    max_v = np.max(source_data)
    return (source_data - min_v) / (max_v - min_v), min_v, max_v

def confidenceInterval(source_data):
    """置信区间"""
    # 假设置信度为 1- α= 95 %
    # 1. 先求样本均值 2. 样本方差
    # StandardDeviation_sum = 0
    # meanvalue = np.mean(source_data)
    # size = len(source_data)
    # dataset = np.asarray(source_data)
    # Sumdata = sum(dataset)
    # meanvalue1 = Sumdata /size
    # for index in dataset:
    #     StandardDeviation_sum = StandardDeviation_sum + (index - meanvalue) ** 2
    # StandardDeviation_sum = StandardDeviation_sum /Sumdata
    # StandardDeviationOfData = StandardDeviation_sum ** 0.5
    #
    # std = np.std(source_data)
    # a = meanvalue - 1.96 * std
    # b = meanvalue + 1.96 * std
    # a1 = meanvalue1 - 1.96 * StandardDeviationOfData
    # b1 = meanvalue1 + 1.96 * StandardDeviationOfData
    # return  a, b, a1 , b1
    StandardDeviation_sum = 0
    # 返回样本数量
    Sizeofdata = len(source_data)
    data = np.array(source_data)
    print(data)
    Sumdata = sum(data)
    # 计算平均值
    Meanvalue = Sumdata / Sizeofdata
    # print(Meanvalue)
    # 计算标准差
    for index in data:
        StandardDeviation_sum = StandardDeviation_sum + (index - Meanvalue) ** 2
    StandardDeviation_sum = StandardDeviation_sum / Sizeofdata
    StandardDeviationOfData = StandardDeviation_sum ** 0.5
    # print(StandardDeviationOfData)
    # 计算置信区间
    LowerLimitingValue = Meanvalue - 1.645 * StandardDeviationOfData
    UpperLimitingValue = Meanvalue + 1.645 * StandardDeviationOfData
    return LowerLimitingValue, UpperLimitingValue


def recovery(normalize_data, min_v , max_v):
    """
    数据还原
    :param normalize_data:
    :return:
    """
    source_data = normalize_data * (max_v - min_v) + min_v
    return source_data


def absolutePercentError(predict_value, real_value):
    if not isinstance(predict_value, np.ndarray) or not isinstance(real_value, np.ndarray):
        predict_value = np.asarray(predict_value)
        real_value = np.asarray(real_value)
    abs_value = np.abs((predict_value - real_value)/real_value)
    error = np.sum(abs_value) / len(real_value)
    return error

def absoluteError(predict_value, real_value):
    """
    每个样本对应的误差
    :param predict_value:
    :param real_value:
    :return:
    """
    if not isinstance(predict_value, np.ndarray) or not isinstance(real_value, np.ndarray):
        predict_value = np.asarray(predict_value)
        real_value = np.asarray(real_value)
    abs_value = np.abs((predict_value - real_value) / real_value)
    return abs_value


def isolation_data (source_data, split_rate=0.6):
    """
     将数据集分为 两部分， 一部分用来训练，一部分用来检测
    :param sorce_data: 原始数据
    :param split_rate: 分割率
    :return:
    """
    size= len(source_data)
    frag = math.ceil(size * split_rate)
    train_data = source_data[:frag]
    test_y = source_data[frag:]
    return train_data, test_y
if __name__ == '__main__':

    # pd.fix_data()
    iris = load_iris()
    data = iris.data
    print(fix_data(data, method="NewtonMethod"))
    # print(pd.Standard([22.0, 14.0, 20.0, 10.0, 12.0, 12.0, 11.0, 20.0, 15.0, 20.0]))