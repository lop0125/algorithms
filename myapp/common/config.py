# -*- coding:utf-8 -*-
"""
@author:xuyi
@time:2018/8/10  10:05
"""
# coding=utf-8
import configparser
import os


class ReadConfigFile():

    def get_value(self):
        root_dir = os.path.dirname(os.path.abspath('.'))  # 获取项目根目录的相对路径
        print(root_dir)
        config = configparser.ConfigParser()
        file_path = root_dir + '/myapp/properties/config.ini'
        print(file_path)
        if os.path.exists(file_path):
            config.read(file_path)
            host = config.get("args", "host")
            port = config.get("args", "port")
            debug = config.get("args", "debug")

            return host, port, debug  # 返回的是一个元组
        else:
            return "文件不存在"


if __name__ == "__main__":
    read = ReadConfigFile()
    print(read.get_value())


