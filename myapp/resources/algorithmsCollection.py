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
import myapp.resources.SVR_GA as svr_ga
parser = reqparse.RequestParser()
#  神经网络参数
parser.add_argument("input_num")  # 神经网络输入神经元个数
parser.add_argument("hidden_num")  # 神经网络隐藏神经元个数
parser.add_argument("output_num")  # 输出层个数
parser.add_argument("algorithms")  # 算法名称
parser.add_argument("algorithm_neural")  # 神经网络灰色预测残差补偿中神经网络参数 ("ANN", "RNN")
parser.add_argument("data")  # 公共参数
parser.add_argument("prediction_num")  # 灰色预测模块参数
# parser.add_argument("require")  # 这个参数决定是否是训练还是进行预测 (train, forecast) 目前要求训练，预测一起，所以这个参数作废
parser.add_argument("xgb_params")
parser.add_argument("num_rounds") # 训练次数
parser.add_argument("itemid_list") # 预测项参数 10001 ~ 10005
parser.add_argument("powerunitid_list")  # 用电单元id 集合



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
        elif algorithm_name == 'svr_ga':
            return jsonify(svr_ga.main(args))

    # def get(self):
    #     return jsonify("xy")

