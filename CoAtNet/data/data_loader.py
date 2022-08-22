import os
import numpy as np
import pandas as pd
import atd2022

import torch
from torch.utils.data import Dataset, DataLoader

class atdDataset():
    def __init__(self, df:pd.DataFrame):

        self.df=df
        self.__read_data__()
    
    def __read_data__(self):
        df = self.df
        self.data = df.values
        #print("data_dim", self.data.shape)

    def __getitem__(self, index):
        #if index >= self.df.shape[0]-20:
        
        #print(index)
        begin = index
        train_x = self.data[begin:begin+20]
        #print(train_x)
        train_y = self.data[begin+20:begin+20+1]
        #print("trainXY_shape", train_x.shape, train_y.shape)
        return train_x, train_y
    
    def __len__(self):
        return len(self.data)-20


class atd_Pred():

    def __init__(self, df:pd.DataFrame):

        self.df=df
        self.__read_data__()
    
    def __read_data__(self):
        df = self.df
        border1=len(df)-20
        border2=len(df)
        data = df.values
        data = data[border1:border2]
        self.data=data

    def __getitem__(self, index):
        begin = index
        train_x = self.data[index, index+20]
        train_y = self.data[index+20+1]
        return train_x, train_y
    
    def __len__(self):
        return len(self.data)
