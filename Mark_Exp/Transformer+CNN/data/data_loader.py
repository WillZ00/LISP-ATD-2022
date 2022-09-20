import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import gc
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class atd_dataset(Dataset):
    def __init__(self, df:pd.DataFrame, history_len=52, predict_len=4):
        self.df=df
        self.history_len = history_len
        self.predict_len = predict_len
        self.__read_data__()


    def __len__(self):
        return len(self.data)-self.history_len-self.predict_len+1
        #return 20

    def __read_data__(self):
        df = self.df
        self.data = df.values
        self.data_1 = df.sort_index(axis=1, level=1).values

    
    def __getitem__(self,idx):
        df = self.df
        history_len = self.history_len
        predict_len = self.predict_len
        
        begin = idx
        train_x = self.data[begin : begin+history_len]
        train_x1 = self.data_1[begin : begin+history_len]
        train_y = self.data[begin+history_len : begin+history_len+predict_len]

        #print("check input dim", train_x.shape, train_y.shape)

        return train_x, train_x1, train_y


class atd_Pred(Dataset):
    def __init__(self, df:pd.DataFrame, history_len=52):
        self.df=df
        self.history_len = history_len
        self.__read_data__()

    def __len__(self):
        return 1

    def __read_data__(self):
        df = self.df
        self.data = df.values
        self.data_1 = df.sort_index(axis=1, level=1).values

    def __getitem__(self,idx):
        df = self.df
        history_len = self.history_len
        #print("history_len", history_len)
        pred_x = self.data[ - history_len :]
        pred_x1 = self.data_1[- history_len : ]
        #print("check input shape", pred_x.shape)

        return pred_x, pred_x1