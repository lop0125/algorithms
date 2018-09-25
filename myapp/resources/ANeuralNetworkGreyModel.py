# -*- coding:utf-8 -*-
"""
@author:xuyi
@time:2018/8/8  8:39
"""
import numpy as np
from GreyModel import GreyModelMethod as gy
from NeuralNetwork import NeuralNetwork as nn
from flask import Flask, jsonify, abort, request
from datetime import datetime
from flask_restful import reqparse, abort, Api, Resource

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument("data")


def min_max_normalize(dataset):
    """
    数据归一化
    :param dataset: 待处理数据
    :return: 归一化之后的数据
    """
    min_v = np.min(dataset)
    max_v = np.max(dataset)
    return (dataset - min_v) / (max_v - min_v), min_v, max_v


def reduction(normalize_data, min_v, max_v):
    """
    数据从归一化恢复
    :param dataset:
    :return:
    """
    source_data = normalize_data * (max_v - min_v) + min_v
    return source_data


class NeuralNetworkGreyModel(Resource):
    def post(self):
        args = parser.parse_args()
        data = args['data']
        data = eval(data)
        dataset = data['value']
        grey_model = gy()  # 实例化灰色预测类
        grey_result = grey_model.start(data=dataset)  # 启动灰色预测进行预测
        dataset = np.asarray(dataset)  # ? 为什么不能 ndarray
        error = dataset - grey_result  # 得到残差
        neuralnetwork_model = nn()  # 神经网络进行实例化
        correction_error, error_min, error_max = neuralnetwork_model.launch(1, 10, dataset, error)  # 启动神经网络残差补偿，得到最终预测结果
        predict = dataset - (error * (error_max - error_min) + error_min)
        my_dict = {"message": "success", "result": predict.tolist(), "date": datetime.now()}
        return jsonify(my_dict)


api.add_resource(NeuralNetworkGreyModel, '/neuralnetworkgreymodel')


if __name__ == '__main__':
    # a = NeuralNetworkGreyModel()
    # print(a.post())
    app.run(debug=True)

