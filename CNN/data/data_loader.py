import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import gc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class atd_dataset(Dataset):
    def __init__(self, df:pd.DataFrame, history_len=52):
        self.df=df
        self.history_len = history_len
        self.__read_data__()


    def __len__(self):
        return len(self.data)-self.history_len-1

    def __read_data__(self):
        df = self.df
        self.data = df

    
    def __getitem__(self,idx):
        df = self.df
        history_len = self.history_len
        
        begin = idx
        train_x = self.data[begin:begin+history_len]
        train_y = self.data[begin+history_len:begin+history_len+1]

        return train_x, train_y


class atd_Pred(Dataset):
    def __init__(self, df:pd.DataFrame, history_len=52):
        self.df=df
        self.history_len = history_len
        self.__read_data__()

    def __len__(self):
        return 1

    def __read_data__(self):
        df = self.df
        self.data = df

    def __getitem__(self,idx):
        df = self.df
        history_len = self.history_len
        
        begin = idx
        pred_x = self.data[begin-history_len:begin]

        return pred_x