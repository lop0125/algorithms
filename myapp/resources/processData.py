# -*- coding:utf-8 -*-
"""
@author:xuyi
@time:2018/9/20  11:41
"""
from flask import Flask, jsonify, abort, request
from flask_restful import reqparse, abort, Api, Resource
import myapp.dataDeal.dealUtil as du

parser = reqparse.RequestParser()
parser.add_argument("data")  # 存放数据
parser.add_argument("way") # 处理数据进行的方法
parser.add_argument("method")
class DealData(Resource):
    def post(self):
        args = parser.parse_args()
        data = args ['data']
        data = eval(data)
        dataset = data['value']
        way = args['way']
        if way == "fix_data":
            method = args ['method']
            return jsonify(du.fix_data(dataset,method= method).tolist())

        elif way == "mean":
            return

