# -*- coding:utf-8 -*-
"""
@author:xuyi
@time:2018/8/9  10:27
"""
from flask import Flask
from flask_restful import Api
from myapp.resources.algorithmsCollection import Algorithms
from myapp.common.config import ReadConfigFile
from myapp.resources.processData import DealData
app = Flask(__name__)
api = Api(app)


api.add_resource(Algorithms, '/algorithms')
api.add_resource(DealData, '/dealData')


if __name__ == '__main__':
    rcf = ReadConfigFile()
    host, port, debug = rcf.get_value()
    app.run(host=host, port=port, debug=bool(debug))

