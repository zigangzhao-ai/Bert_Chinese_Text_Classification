"""
functions:转换xlsx到tsv/csv文件
refer to: https://blog.csdn.net/qq_44956991/article/details/124749602
"""

import os
import glob
import pandas as pd

xlsx_pths = glob.glob("/dataset/0629.xlsx") #文件目录，包含xlsx文件
for xlsx_pth in xlsx_pths: 
    data = pd.read_excel(xlsx_pth) #此时获得的data 是dataframe结构，即键对应的属性结构
    #将data中列名为'label', 'text'的2列筛选出来存到p
    # print('--', data)
    p = data.loc[:, ['label', 'text']]
    for i in range(len(p)):
        txt = p.loc[i, ['text']].str.replace('\r', ' ') 
        txt = pd.Series(txt)
        p.loc[i, ['text']] = txt
        # print('---',  p.loc[i, ['text']])
    #列名为'text'更改为'text_a'
    p.rename(columns={'text': 'text_a'}, inplace=True) 
    #portion为名称和后缀分离后的列表
    portion = os.path.splitext(xlsx_pth)
    #批量生成对原来读取的.xlsx的文件名+'.tsv'格式/'.csv'
    p.to_csv(portion[0]+'.tsv', sep="\t", index=False)