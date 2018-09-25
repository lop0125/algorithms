# -*- coding:utf-8 -*-
"""
@author:xuyi
@time:2018/8/9  11:23
"""
import tensorflow as tf
import numpy as np
import myapp.resources.greymodelimplementation as greymodel
from myapp.resources.GreyModel2 import GreyModel2
from myapp.resources.neuralnetworkimplementation import NeuralNetwork
from flask import Flask, jsonify, abort, request
from flask_restful import reqparse, abort, Api, Resource
import myapp.resources.NNGMimplemention as nngy
import myapp.resources.XgboostAlgorithms as xgbAlgorithms
parser = reqparse.RequestParser()
#  神经网络参数
parser.add_argument("input_num")  # 神经网络输入神经元个数
parser.add_argument("hidden_num")  # 神经网络隐藏神经元个数
parser.add_argument("output_num")  # 输出层个数
parser.add_argument("algorithms")  # 算法名称
parser.add_argument("algorithm_neural")  # 神经网络灰色预测残差补偿中神经网络参数 ("ANN", "RNN")
parser.add_argument("data")  # 公共参数
parser.add_argument("predict_num_year")  # 灰色预测模块参数

parser.add_argument("xgb_params")



"""========================= 预测算法接口 =========================
"""


class Algorithms(Resource):
    def post(self):
        args = parser.parse_args()
        algorithm_name = args['algorithms']
        if algorithm_name == 'RNN' or algorithm_name == 'ANN':
            neuralnetwork = NeuralNetwork()
            return jsonify(neuralnetwork.launch(args))
        elif algorithm_name == 'gerymodel':
            return jsonify(greymodel.launch(args))
        elif algorithm_name == 'gerymodel2':
            greymodel2 = GreyModel2()
            return jsonify(greymodel2.launch(args))
        elif algorithm_name == 'residual_compensation':   # 残差补偿
            return jsonify(nngy.launch(args))
        elif algorithm_name == 'xgboost':
            return jsonify(xgbAlgorithms.launch(args))
