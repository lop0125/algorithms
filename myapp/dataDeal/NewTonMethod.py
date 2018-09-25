# -*- coding:utf-8 -*-
"""
@author:xuyi
@time:2018/9/5  17:27
"""
import matplotlib.pyplot as plt
class NewtonInterpolation():

    """
    牛顿插值法
    插值的函数表为
    xi      0.4，       0.55，     0.65，      0.80，       0.90，   1.05
    f(xi)   0.41075,    0.57815,   0.69675,    0.88811,    1.02652,  1.25382
    """
    def difference_quotient(self, x, y):
        """
        计算差商的值
        :param x:
        :param y:
        :return:
        """
        # i记录计算差商的次数，这里循环5次，计算5次差商。
        n = len(x)
        i = 0
        quotient = [0] * (n+1)
        while i < n - 1:
            j = n - 1
            while j > i:
                if i == 0:
                    quotient[j] = ((y[j] - y[j - 1]) / (x[j] - x[j - 1]))
                else:
                    quotient[j] = (quotient[j] - quotient[j - 1]) / (x[j] - x[j - 1 - i])
                j -= 1
            i += 1
        return quotient


    def functionExpression(self, unkown_x, x, y, parameters):
        temp = 1
        express = y[0]
        for i in range(1, len(parameters)):
            temp *= (unkown_x - x[i-1])
            express += parameters[i] * temp
        return express

        # return y[0] + parameters[1] * (unkown_x - x[0]) + parameters[2] * (unkown_x - x[0]) * (unkown_x - x[1]) + \
        #        parameters[3] * (unkown_x - x[0]) * (unkown_x - x[1]) * (unkown_x - x[2]) \
        #        + parameters[4] * (unkown_x - x[0]) * (unkown_x - x[1]) * (unkown_x - x[2])*(unkown_x - x[3])\
        #        + parameters[5] * (unkown_x - x[0]) * (unkown_x - x[1]) * (unkown_x - x[2])*(unkown_x - x[3]) \
        #        * (unkown_x - x[4])


    """计算插值多项式的值"""


    def calculate_data(self,unkown_x_list, x, y):
        parameters = self.difference_quotient(x, y)
        returnData = []
        if isinstance(unkown_x_list,list):
            for unkown_x in unkown_x_list:
                returnData.append(self.functionExpression(unkown_x, x, y, parameters))
        else:
            returnData.append(self.functionExpression(unkown_x_list, x, y, parameters))
        return returnData


    """画函数的图像
    newData为曲线拟合后的曲线
    """
    # def draw(self, newData):
    #     plt.scatter(x, y, label="离散数据", color="red")
    #     plt.plot(x, newData, label="牛顿插值拟合曲线", color="black")
    #     plt.scatter(0.596, function(0.596), label="预测函数点", color="blue")
    #     plt.title("牛顿插值法")
    #     mpl.rcParams['font.sans-serif'] = ['SimHei']
    #     mpl.rcParams['axes.unicode_minus'] = False
    #     plt.legend(loc="upper left")
    #     plt.show()


    # parameters = five_order_difference_quotient(x, y)
    # yuanzu = calculate_data(x, parameters)
    # #

if __name__ == "__main__":
    nt = NewtonInterpolation()
    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # y = [4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9]
    # # x = [0.4, 0.55, 0.65, 0.80, 0.90, 1.05]
    # y = [0.41075, 0.57815, 0.69675, 0.88811, 1.02652, 1.25382]
    x = [1, 2, 3, 4, 5, 6]
    y = [1, 4, 9, 16, 25, 36]
    plt.scatter(x, y, label="离散数据", color="red")
    newData = nt.calculate_data(7.5,x, y)
    plt.plot(x, y, label="牛顿插值拟合曲线", color="black")
    plt.scatter(7, nt.calculate_data(7,x, y), label="预测函数点", color="blue")
    plt.show()