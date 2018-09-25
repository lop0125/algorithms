# -*- coding:utf-8 -*-
"""
@author:xuyi
@time:2018/9/10  14:17
"""
import os
import configparser
import mysql.connector as sqlconnect
# print(os.path.dirname(os.path.abspath("."))+ "\\properties\jdbc.ini")
def getConfigArgs():
    dirpath = os.path.dirname(os.path.abspath("."))
    config = configparser.ConfigParser()
    print(type(config))
    configPath = dirpath + "\\properties\jdbc.ini"
    if os.path.exists(configPath):
        config.read(configPath)
        host = config.get("jdbc","host")
        user = config.get("jdbc", "user")
        password = config.get("jdbc", "password")
        database = config.get("jdbc", "database")
        charset = config.get("jdbc", "charset")
        port = config.get("jdbc", "port")
        return host, user, password, database, charset, port
    else:
        print("该文件不存在")

def Mysql_Connet_python():
    host, user, password, database, charset, port = getConfigArgs()
    conn = sqlconnect.connect(host=host,user=user,password=password,database=database,charset=charset,port=port)
    cursor = conn.cursor()
    return cursor, conn

def select_(sql, parmas=None):
    cursor, conn = Mysql_Connet_python()
    try:
        if parmas:
            cursor.execute(sql, parmas)
        else:
            cursor.execute(sql)
        values = cursor.fetchall()
        return values
    except Exception as e:
        print(e)
    finally:
        cursor.close()

def delete_(sql, parmas=None):
    cursor, conn = Mysql_Connet_python()
    try:
        if parmas:
            cursor.execute(sql,parmas)
        else:
            cursor.execute(sql)
    except Exception as e:
        print(e)
    finally:
        cursor.close()


if __name__ == '__main__':
    # sql= "insert into teacher(id, name) value('123', 'xx')"
    # Mysql_Connet_python(sql)
    sql2 = "select * from user where id = %s"
    parmas =("007",)
    print(select_(sql2,parmas))