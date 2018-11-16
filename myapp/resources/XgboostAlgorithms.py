# -*- coding:utf-8 -*-
"""
@author:xuyi
@time:2018/9/3  10:43
"""
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import myapp.dataDeal.dealUtil as du


def launch(args):
    source_data = args['data']
    source_data = eval(source_data)
    predict_num = args['prediction_num']
    xgb_params = args['xgb_params']
    xgb_params = eval(xgb_params)
    itemList = args['itemid_list']
    itemList = eval(itemList)
    itemList = itemList['id']
    input_num = args['input_num']
    input_num = eval(input_num)
    powerunitlist = args['powerunitid_list']
    powerunitlist = eval(powerunitlist)
    powerunitlist = powerunitlist['id']
    allpowerunits = {}
    if predict_num is not None:
        for i in powerunitlist:
            allitems = {}
            i = "_"+i
            data = source_data [i]
            if 10005 in  itemList or '10005' in itemList:
                total_value = data['total_value']  # 总
                dictobj = goForecast(total_value,input_num,predict_num,xgb_params)
                allitems["total"] = dictobj
            if 10001 in  itemList or '10001' in itemList:
                tip_value = data['tip_value']  # 尖
                dictobj = goForecast(tip_value,input_num,predict_num,xgb_params)
                allitems["tip"] = dictobj
            if 10002 in itemList or '10002' in itemList:
                peak_value = data['peak_value']  # 峰
                dictobj = goForecast(peak_value,input_num, predict_num, xgb_params)
                allitems["peak"] =  dictobj
            if 10003 in itemList :
                flat_value = data['flat_value']  # 平
                dictobj = goForecast(flat_value, input_num, predict_num, xgb_params)
                allitems ["flat"]= dictobj
            if 10004 in itemList:
                valey_value = data['valey_value']  # 谷
                dictobj = goForecast(valey_value, input_num, predict_num , xgb_params)
                allitems ["valey"]= dictobj
            allpowerunits[i] = allitems

        return allpowerunits
    else :
        return {"status":"1",
                "message": "错误！请输入想要预测的单位时间"}





def goForecast (source_data , input_num, prediction_num,xgb_params):
    source_data = np.asarray(source_data, dtype=float)
    source_data, min_v, max_v = du.min_max_normalize(source_data)
    """ 制作数据集，前面input_num个数作为因子，后面一个数作为输出
    """
    if input_num is not None:
        input_num = int(input_num)
        dataset, train_data, label_data = du.create_dataset(source_data, input_num)
    else:
        dataset, train_data, label_data = du.create_dataset(source_data)

    # iris = load_iris()  # 进行测试的数据集
    # train_data = iris.data
    # label_data = iris.target
    """
    将数据集分为训练集跟样本集
    """
    X_train, X_test, y_train, y_test = train_test_split(train_data, label_data, test_size=0.25, random_state=0)
    dtrain = xgb.DMatrix(X_train, y_train)
    print(type(dtrain))
    num_rounds = 1000
    #  训练参数
    model = xgb.train(xgb_params, dtrain, num_rounds)
    model.dump_model('dump.raw.txt')
    # 对测试集预测
    # dtest = xgb.DMatrix(X_test)
    # ans = model.predict(dtest)
    #  require = args ['require']  # 对数据集是训练还是预测

    """============================训练==========================="""
    train_value = model.predict(xgb.DMatrix(train_data))

    """==========================预测=========================="""


    predict_num = int(prediction_num)
    train_value_copy = label_data[:].tolist()
    # source_data_copy = source_data_copy.tolist()
    for i in range(predict_num):
        temp_arr = []
        temp_arr.append(train_value_copy[(len(label_data) + i - input_num):])
        result = model.predict(xgb.DMatrix(temp_arr))
        train_value_copy.append(result[0])
    predict_value = train_value_copy[-(predict_num):]
    # predict_value = du.recovery(np.asarray(predict_value), min_v, max_v)
    predict_value = np.asarray(predict_value)
    train_value = du.recovery(train_value, min_v, max_v)  # 归一化还原数据
    label_value = du.recovery(label_data, min_v, max_v)
    # print(predict_data, label_data , len(predict_data))
    error = np.fabs((train_value - label_value) / label_value)
    # np.set_printoptions(precision=2)
    count = 0
    a, b = du.confidenceInterval(label_value)
    for i in range(len(train_value)):
        if train_value[i] > a and train_value[i] < b:
            print(train_value[i], a, b)
            count += 1
    byxs = du.coefficientOfDispersion(train_value, axis=0)
    print(byxs)
    mydict = {"message": "success",
              # "status": "0",
              # "result": train_value.tolist(),
              # "label": label_value.tolist(),
              # "error": error.tolist(),
              "predict_value": predict_value.tolist()
              # "byxs": str(byxs)z
             }
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

# except Exception as e:
# return {"message": "task is failed, beacuse " + str(e),
#         "status": "1"}
    pass