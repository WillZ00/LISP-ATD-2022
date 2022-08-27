import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import gc
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time



class atd_dataset(Dataset):
    def __init__(self, df:pd.DataFrame, lispStats, history_len=52):
        self.df=df
        self.history_len = history_len
        self.lispStats = lispStats
        self.data = df.values
        self.__read_data__()


    def __len__(self):
        return len(self.data)-self.history_len-1

    def __read_data__(self):
        # df = self.df
        self.data = self.df.values
        #self.stats_pred = self.lispStats.predict(self.lispStats._fit_processing(self.df))
    
    def __getitem__(self,idx):
        history_len = self.history_len
        
        begin = idx
        train_x = self.data[begin : begin+history_len]
        truth = self.data[begin+history_len : begin+history_len+1]
        #print("check dtype", (self.data[:begin+history_len]).shape)
        #print("hello")
        start_time = time.time()
        stats_pred = self.lispStats.predict(self.lispStats._fit_processing(self.data[:begin+history_len], flag='test'))
        print(time.time()-start_time)
        train_y = truth-stats_pred

        #print("check input dim", train_x.shape, train_y.shape)

        return train_x, train_y


class atd_Pred(Dataset):
    def __init__(self, df:pd.DataFrame, lispStats, history_len=52):
        self.df=df
        self.history_len = history_len
        self.lispStats = lispStats
        self.__read_data__()

    def __len__(self):
        return 1

    def __read_data__(self):
        df = self.df
        self.data = df.values

    def __getitem__(self,idx):
        df = self.df
        history_len = self.history_len
        #print("history_len", history_len)
        begin = len(self.data)
        pred_x = self.data[begin - history_len : begin]
        #stats_pred = self.lispStats.predict(self.data)
        stats_pred = self.lispStats.predict(self.lispStats._fit_processing(self.data[:begin], flag='test'))
        #print("check input shape", pred_x.shape)

        return pred_x, stats_pred