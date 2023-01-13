import pandas as pd
import os

data = pd.read_csv('mfeng/LISP-ATD-2022/Mark_Exp/test new data/web_traffic/data/train_1.csv', index_col=0)
os.getcwd()
os.listdir('mfeng')
print(data)
data.columns
data = data.set_index('Page')
