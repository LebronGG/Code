import numpy as np
import pandas as pd
from sklearn import preprocessing
import xlrd
 
def excel_to_matrix(path):
    table = xlrd.open_workbook(path).sheets()[0]#获取第一个sheet表
    row = table.nrows  # 行数
    col = table.ncols  # 列数
    datamatrix = np.zeros((row, col))#生成一个nrows行ncols列，且元素均为0的初始矩阵
    for x in range(col):
        cols = np.matrix(table.col_values(x))  # 把list转换为矩阵进行矩阵操作
        datamatrix[:, x] = cols # 按列把数据存进矩阵中
    #数据归一化   
    #min_max_scaler = preprocessing.MinMaxScaler()
    #datamatrix  = min_max_scaler.fit_transform(datamatrix)
    return datamatrix
 
datafile = u'1.xlsx'
a=excel_to_matrix(datafile)
np.savetxt('new.csv', a, delimiter = ',')


