# -*- coding:utf-8 -*-
"""
@author:xuyi
@time:2018/8/9  12:15
"""
from myapp.resources.GreyModel import GreyModelMethod
from myapp.resources.NeuralNetwork import NeuralNetwork
import numpy as np

from datetime import datetime


def launch(parser):
    try:
        data = parser['data']
        data = eval(data)
        dataset = data['value']
        grey_model = GreyModelMethod()  # 实例化灰色预测类
        grey_result = grey_model.start(data=dataset)  # 启动灰色预测进行预测
        dataset = np.asarray(dataset)  # ? 为什么不能 ndarray
        error = dataset - grey_result  # 得到残差
        neuralnetwork_model = NeuralNetwork()  # 神经网络进行实例化
        input_num = 1  # 目前输入神经元固定为一个
        hidden_num = parser['hidden_num']
        hidden_num = int(hidden_num)
        output_num = parser['output_num']
        algorithm_neural = parser['algorithm_neural']
        if output_num is not None:
            correction_error, error_min, error_max = neuralnetwork_model.launch(input_num, hidden_num, dataset,
                                                                                error, no=output_num,
                                                                                algorithm_neural=algorithm_neural)
        else:
            # 启动神经网络残差补偿，得到最终预测结果
            correction_error, error_min, error_max = neuralnetwork_model.launch(input_num, hidden_num, dataset,
                                                                                error)

        predict = dataset - (error * (error_max - error_min) + error_min)
        my_dict = {"message": "success", "result": predict.tolist(), "date": datetime.now()}
        return my_dict
    except Exception as e:
        return {"message": "failed:" + str(e)}