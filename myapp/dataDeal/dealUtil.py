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

#自定义列向量插值函数
#s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
def ployinterp_column(s, n, method="lagrange", k=5):
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

def fix_data(data, method="mean"):
    """
    数据进行修复处理
    :param data: 原始数据
    :param n: 不正常数据的位置
    :param method: 进行数据修复的方法，有均值，取中值，拉格朗日插值，牛顿插值法
    :return: 进行修复过的数据
    """
    n = 10
    if isinstance(data, list):
        data = np.asarray(data)
    # 制作一份副本
    X_t = data.copy()
    # 在第二行制作一些丢失值
    X_t[n,:] = np.repeat("NaN", data.shape[1])
    # print(X_t)
    # 修补数据
    if method == "mean" or method == "median" or method == "most_frequent":
        # 创建一个imputer 对象，采用平均值策略
        imputer = Imputer(missing_values="NaN", strategy= method)
        X_imputer = imputer.fit_transform(X_t)
        return X_imputer
    elif method == 'lagrange' or method == 'NewtonMethod':
        for i in range(4):
             for j, value in enumerate (X_t[:, i]):
                if value != value:
                    X_t[j, i] = ployinterp_column(X_t[:, i], j ,method=method)
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

def coefficientOfDispersion(self,a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue, overwrite_input=False):
    """变异系数"""
    return self.getStandardDeviation(a, axis, dtype, out, ddof, keepdims) / self.getMean(a, axis, out, overwrite_input, keepdims)

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


def recovery(normalize_data, min_v , max_v):
    """
    数据还原
    :param normalize_data:
    :return:
    """
    source_data = normalize_data * (max_v - min_v) + min_v
    return source_data


if __name__ == '__main__':

    # pd.fix_data()
    iris = load_iris()
    data = iris.data
    print(fix_data(data, method="NewtonMethod"))
    # print(pd.Standard([22.0, 14.0, 20.0, 10.0, 12.0, 12.0, 11.0, 20.0, 15.0, 20.0]))