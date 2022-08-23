import os
import numpy as np
import pandas as pd
import atd2022

import torch
from torch.utils.data import Dataset, DataLoader

class atdDataset():
    def __init__(self, region_name:str, df:pd.DataFrame):
        self.region_name=region_name
        self.df=df
        self.__read_data__()
    
    def __read_data__(self):
        df = self.df

        #df = df.iloc[:-20].stack(level=0).sort_index(level=1)
        #print("check input df shape", df.shape)
        self.data = df.loc[(slice(None), self.region_name),:].iloc[:-20].values
        #print("data_dim", self.data.shape)

    def __getitem__(self, index):
        #if index >= self.df.shape[0]-20:
        
        #print(index)
        begin = index
        train_x = self.data[begin:begin+20]
        train_x, mean, std = self._normalize_data(train_x)
        #print(train_x)
        train_y = self.data[begin+20:begin+20+1]

        train_y = (train_y - mean)/std
        # train_y = np.array([(train_y[i]-mean[i])/std[i] if std[i] !=0 else train_y[i]-mean[i] for i in range(len(train_y))])
        #print("params",mean, std)
        #print("trainXY_shape", train_x.shape, train_y.shape)
        #print("check input x shape", train_x.shape)
        return train_x, train_y

    
    def _normalize_data(self, data):
        data_mean = np.average(data, axis=1)
        data_std = np.std(data, axis=1)
        data_std[data_std==0]=1
        data_normalized = (data - data_mean)/data_std
        return data_normalized, data_mean, data_std
            
    
    def __len__(self):
        return len(self.data)-20-1


class atd_Pred():

    def __init__(self, region_name, df:pd.DataFrame):
        self.region_name = region_name
        self.df=df
        self.__read_data__()
        #self.region_name = region_name
  

    def __read_data__(self):
        df = self.df
        self.data = df.loc[(slice(None), self.region_name),:].values

    def __getitem__(self, index):
        #if index >= self.df.shape[0]-20:
        
        #print(index)
        begin = len(self.data)
        pred_x = self.data[begin-20:begin]
        pred_x, mean, std = self._normalize_data(pred_x)
        #print(train_x)
        #train_y = self.data[begin+20:begin+20+1]
        #print("trainXY_shape", train_x.shape, train_y.shape)
        return pred_x, mean, std

    def _normalize_data(self, data):
        data_mean = np.average(data, axis=1)
        data_std = np.std(data, axis=1)
        data_std[data_std==0]=1
        data_normalized = (data - data_mean)/data_std
        return data_normalized, data_mean, data_std

    
    def __len__(self):
        return 1

