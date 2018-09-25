# -*- coding:utf-8 -*-
"""
@author:xuyi
@time:2018/9/3  10:43
"""
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_iris
import myapp.dataDeal.dealUtil as du


def launch(args):
    data = args['data']
    data = eval(data)
    source_data = data['value']
    source_data, min_v, max_v = du.min_max_normalize(source_data)
    input_num = args['input_num']
    if input_num is not None:
        input_num = int(input_num)
        dataset, train_data, label_data = du.create_dataset(source_data, input_num)
    else:
        dataset, train_data, label_data = du.create_dataset(source_data)
    xgb_params = args['xgb_params']
    xgb_params = eval(xgb_params)
    # iris = load_iris()  # 进行测试的数据集
    # train_data = iris.data
    # label_data = iris.target
    X_train, X_test, y_train, y_test = train_test_split(train_data, label_data, test_size=0.2, random_state=0)
    dtrain = xgb.DMatrix(X_train, y_train)
    print(type(dtrain))
    num_rounds = 1000
    #  训练参数
    model = xgb.train(xgb_params, dtrain, num_rounds)

    # 对测试集预测
    dtest = xgb.DMatrix(X_test)
    ans = model.predict(dtest)
    predict_data = model.predict(xgb.DMatrix(train_data))
    predict_data = du.recovery(predict_data, min_v, max_v)
    label_data = du.recovery(label_data, min_v, max_v)
    print(predict_data, label_data , len(source_data))
    mydict = {"message": "success",
               "result": predict_data.tolist(),
                "label": label_data.tolist()}
    return mydict
    # print("ans", ans)
    # print(y_test)
    # # 判断是否准确
    # count1 = 0
    # count2 = 0
    # for i in range(len(y_test)):
    #     if ans[i] == y_test[i]:
    #         count1 += 1
    #     else:
    #         count2 += 1
    # print(count1 / (count1 + count2))
    # if (count1 / (count1 + count2)) > 0.95:
    #     return "可以用来预测"
    # else:
    #     return "不能用来预测"
