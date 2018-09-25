# -*- coding:utf-8 -*-
"""
@author:xuyi
@time:2018/9/6  14:43
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
filepath = os.path.abspath(os.path.join(os.getcwd(),".."))
filepath = filepath + '\\data\发远光电量2.xlsx'
dataframe = pd.read_excel(filepath)
# print (dataframe.iloc[:,2])
# print(dataframe.columns.values[2])


def returnMonthData(n):
    """
     得到某一月的数据
    :param n: 月份
    :return:
    """
    if n <= 0:
        print("n必须大于0")
        return
    else:
        temp_df = dataframe.iloc[:, n]
        month = dataframe.columns.values[n]
        t = temp_df.values
        return month, t

def getEachMonth(n):
    if n <=0:
        print("n 必须大于0")
        return
    else:
        temp_df = dataframe.iloc[n-1,1:]
        index = dataframe.iloc[n-1][0] # 第几行数据
        return index, temp_df.values

#for i in range(1,13):

# x_month = dataframe.columns.values[1:]
# x_indexs = []
*_,Y = getEachMonth(3)
# plt.figure()
# plt.plot(x_month, Y)
# plt.show()
print(type(Y.shape))