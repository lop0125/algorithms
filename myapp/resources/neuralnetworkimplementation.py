# -*- coding:utf-8 -*-
"""
@author:xuyi
@time:2018/8/9  12:02
"""
import numpy as np
import tensorflow as tf
from datetime import datetime
"""===============================神经网络模块======================================
"""


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
    # def create_dataset(self, data, look_back=4):
    #     X = []
    #     for i, v in enumerate(data[look_back:], look_back):
    #         X += [data[i - look_back:i + 1]]
    #     return np.asarray(X)

    def RNN(self):
        input_weights = tf.Variable(tf.random_normal([self.input_num, self.hidden_num]))
        input_bias = tf.Variable(tf.random_normal([1]))
        hidden_weights = tf.Variable(tf.random_normal([self.hidden_num, self.output_num]))
        hidden_bias = tf.Variable(tf.random_normal([1]))

        x = tf.placeholder(dtype=tf.float32, shape=(None, self.input_num), name='input')
        hidden_cells = tf.tanh(tf.matmul(x, input_weights) + input_bias)
        out = tf.tanh(tf.matmul(hidden_cells, hidden_weights) + hidden_bias)
        return out, x

    def ANN(self):
        weights = tf.Variable(tf.random_normal([self.hidden_num, 1]))
        bias = tf.Variable(tf.random_normal([1]))

        x = tf.placeholder(dtype=tf.float32, shape=(None, self.input_num), name='input')
        unit = tf.nn.rnn_cell.GRUCell(self.hidden_num, name='GRUCell')
        output, status = tf.nn.static_rnn(unit, [x], dtype=tf.float32)
        out = tf.matmul(output[-1], weights) + bias
        return out, x

    def launch(self, parser):
        try:
            ni = parser['input_num']
            nh = parser['hidden_num']
            no = parser['output_num']
            data = parser['data']
            algorithm = parser['algorithms']
            self.setup(int(ni), int(nh))
            data = eval(data)
            dataset = data['value']
            # dataset = pandas.read_csv('airline_passengers.csv',
            #                       usecols=[1])

            dataset = np.asarray(dataset)
            dataset = dataset[np.nonzero(dataset == dataset)]  # 找出空值NaN  因为Nan！= Nan
            dataset, dataset_min, dataset_max = min_max_normalize(dataset)
            ds = self.create_dataset(dataset[:])  # 除去最后一个值
            size = len(ds)
            frag = int(size * 0.6)
            train_ds = ds[:frag, :]
            test_ds = ds[frag:, :]
            if algorithm == "ANN":
                out, x = self.ANN()
            elif algorithm == "RNN":
                out, x = self.RNN()
            loss, y = cost(out)

            train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
            with tf.Session() as sess:
                init = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())
                sess.run(init)
                batch_size = 10
                for i in range(100):
                    T_x, T_y = train_ds[:, 0:self.input_num], train_ds[:, self.input_num]
                    T_y = np.expand_dims(T_y, -1)

                    for j in np.arange(0, len(T_x), batch_size):
                        loss_v, _ = sess.run([loss, train_op], feed_dict={x: T_x[j:j + batch_size, :],
                                                                          y: T_y[j:j + batch_size, :]})

                T_x, T_y = ds[:, 0:self.input_num], ds[:, self.input_num]
                batch_size = 1

                vs = []
                for i in np.arange(0, len(T_x), batch_size):
                    v = sess.run(out, feed_dict={x: T_x[i:i + batch_size, :]})
                    vs.append(v[0])

                forecast = np.asarray(vs) * (dataset_max - dataset_min) + dataset_min
                my_dict = {"message": "success", "result": forecast.tolist(), "date": datetime.now()}
            return my_dict
        except Exception as e:
            return {"message": "task is failed,the cause is" + str(e.args)}