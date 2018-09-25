# -*- coding:utf-8 -*-
"""
@author:xuyi
@time:2018/8/7  10:38
"""
import tensorflow as tf
import numpy as np
import pandas
from flask import Flask, jsonify, abort, request
from datetime import datetime
from flask_restful import reqparse, abort, Api, Resource


def min_max_normalize(dataset):
    """
    数据归一化
    :param dataset: 待处理数据
    :return: 归一化之后的数据
    """
    min_v = np.min(dataset)
    max_v = np.max(dataset)
    return (dataset - min_v) / (max_v - min_v), min_v, max_v


def cost(out):   # 损失函数
    """
     损失函数
    :param out: 输出集
    :return: 损失函数的值
    """
    y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='true')
    diff = out - y
    loss = tf.reduce_mean(tf.square(diff))
    return loss, y


class NeuralNetwork():

    def __init__(self):
        self.input_num = 0
        self.hidden_num = 0
        self.output_num = 0

    def setup(self, ni, nh, no=1):
        self.input_num = ni
        self.hidden_num = nh
        self.output_num = no

    def create_dataset(self, data):
        """
         # 创建数据集，每五个为一组，每组的第二个数为下一组的第一个数， 前面那四个是输入，最后一个数是输出
        :param data: 待处理数据
        :return: 处理之后的数据
        """
        X = []
        for i, v in enumerate(data[self.input_num:], self.input_num):  # 遍历序列，可以有下标，下标从look_back 开始
            X += [data[i - self.input_num:i + 1]]
        return np.asarray(X)

    def ANN(self):
        input_weights = tf.Variable(tf.random_normal([self.input_num, self.hidden_num]))
        input_bias = tf.Variable(tf.random_normal([1]))
        hidden_weights = tf.Variable(tf.random_normal([self.hidden_num, self.output_num]))
        hidden_bias = tf.Variable(tf.random_normal([1]))

        x = tf.placeholder(dtype=tf.float32, shape=(None, self.input_num), name='input')
        hidden_cells = tf.tanh(tf.matmul(x, input_weights) + input_bias)
        out = tf.tanh(tf.matmul(hidden_cells, hidden_weights) + hidden_bias)
        return out, x

    def RNN(self):
        weights = tf.Variable(tf.random_normal([self.hidden_num, 1]))
        bias = tf.Variable(tf.random_normal([1]))

        x = tf.placeholder(dtype=tf.float32, shape=(None, self.input_num), name='input')
        unit = tf.nn.rnn_cell.GRUCell(self.hidden_num, name='GRUCell')
        output, status = tf.nn.static_rnn(unit, [x], dtype=tf.float32)
        out = tf.matmul(output[-1], weights) + bias
        return out, x
    # BATCH_SIZE 这是定义的一个数量，即一次训练模型投入的样例数
    BATCH_SIZE = 8
    # 生成模拟数据

    def launch(self, ni, nh, dataset, error, no=1, algorithm_neural="ANN"):
        try:
            self.setup(ni, nh)
            dataset = dataset[np.nonzero(dataset == dataset)]  # 找出空值NaN  因为Nan！= Nan
            dataset, *_ = min_max_normalize(dataset)
            error, error_min, error_max = min_max_normalize(error)
            ds = dataset
            if algorithm_neural == "ANN":
                out, x = self.ANN()
            elif algorithm_neural == "RNN":
                out, x = self.RNN()
            loss, y = cost(out)

            train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
            with tf.Session() as sess:
                init = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())
                sess.run(init)
                batch_size = 10  # 10 组数据为一批投入，不然数据量太大容易死机
                for i in range(100):
                    T_x, T_y = ds[:], error[:]
                    T_x = np.expand_dims(T_x, -1)
                    T_y = np.expand_dims(T_y, -1)
                    for j in np.arange(0, len(T_x), batch_size):
                        loss_v, _ = sess.run([loss, train_op], feed_dict={x: T_x[j:j + batch_size, :],
                                                                          y: T_y[j:j + batch_size, :]})

                T_x, T_y = ds[:], error[:]
                T_x = np.expand_dims(T_x, -1)
                batch_size = 1

                vs = []
                for i in np.arange(0, len(T_x), batch_size):
                    v = sess.run(out, feed_dict={x: T_x[i:i + batch_size, :]})
                    vs.append(v[0])

            return np.asarray(vs), error_min, error_max
        except Exception as e:
            return {"message": "fail:" + str(e)}


if __name__ == '__main__':
    nn = NeuralNetwork()
    result = nn.launch(1, 10, dataset=[724.57, 746.62, 778.27, 800.8, 827.75, 871.1, 912.37, 954.28, 995.01, 1037.2]
                        , error=[10, 23, 24, 35, 45, 23, 14, 12, 23, 19], algorithm="ANN")
    print(result)