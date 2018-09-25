# -*- coding:utf-8 -*-
"""
@author:xuyi
@time:2018/8/7  16:58
"""
import math
import numpy as np
from flask import Flask, jsonify, abort, request
from flask_restful import reqparse, abort, Api, Resource

app = Flask(__name__)
api = Api(app)

parse = reqparse.RequestParser()
parse.add_argument("data")
parse.add_argument("predict_num_year")


def predict(num, f, prime_data, a, u):
    for i in range(0, num):
        f[i] = (prime_data[0] - u / a) * (1 - math.exp(a)) * math.exp(-a * (i + num))
    return f


def simulation(num, f, prime_data, a, u):
    for i in range(0, num):
        f[i] = (prime_data[0] - u / a) * (1 - math.exp(a)) * math.exp(-a * i)
    return f


class GreyModelMethod():

    def start(self, data):
        history_data = data

        n = len(history_data)
        # 使用 a^(-x) 来使原数据光滑，a >1 令 a = 2
        X0 = np.array(history_data)

        # 累加生成
        history_data_agg = [sum(history_data[0:i + 1]) for i in range(n)]
        X1 = np.array(history_data_agg)

        B = np.zeros([n - 1, 2])  # n-1 行 2 列 计算数据矩阵B 和数据向量Y  减一的原因是因为第一个值被当做初始值
        Y = np.zeros([n - 1, 1])  # n-1 行 1 列
        #  填充数据
        for i in range(0, n - 1):
            B[i][0] = -0.5 * (X1[i] + X1[i + 1])
            B[i][1] = 1
            Y[i][0] = X0[i + 1]

        # 计算GM(1,1) 微分方程的参数a，u
        # A = np.zeros([2, 1])
        A = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
        a = A[0][0]  # 算出结果是一列， 上面那个是 a ，下面那个是u
        u = A[1][0]

        # 建立灰色预测模型
        XX0 = np.zeros(n)
        XX0[0] = X0[0]
        for i in range(1, n):
            XX0[i] = (X0[0] - u / a) * (1 - math.exp(a)) * math.exp(-a * i)

        # 模型精度的后验差检验
        e = 0
        for i in range(0, n):
            e += (X0[i] - XX0[i])
        e /= n

        # 求历史数据平均值
        aver = 0
        for i in range(0, n):
            aver += X0[i]
        aver /= n

        # 求历史数据方差
        s12 = 0
        for i in range(0, n):
            s12 += (X0[i] - aver) ** 2
        s12 /= n

        # 求残差方差
        s22 = 0
        for i in range(0, n):
            s22 += (X0[i] - XX0[i] - e) ** 2
        s22 /= n

        # 求后验差比值
        C = s22 / s12

        # 求小误差概率
        count = 0
        for i in range(0, n):
            if abs((X0[i] - XX0[i]) - e) < 0.6754 * math.sqrt(s12):
                count = count + 1
            else:
                count = count

        P = count / n

        # if C < 0.35 and P > 0.95:
        # m = 10 + n
        s = np.zeros(n)
        simulation_data = simulation(n, s, X0, a, u)

        return simulation_data
        # else:
        #     return {"message": "该数据集不适合灰色预测"}


if __name__ == '__main__':
    gy = GreyModelMethod()
    result = gy.start(data=[724.57, 746.62, 778.27, 800.8, 827.75, 871.1, 912.37, 954.28, 995.01, 1037.2])
    if type(result) == 'dict':
        print('不适合')
    else:
        print(result)
