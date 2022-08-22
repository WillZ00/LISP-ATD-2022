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
        self.data = df.valuess

    def __getitem__(self, index):
        begin = index
        train_x = self.data[index, index+20]
        train_y = self.data[index+20+1]
        return train_x, train_y
    
    def __len__(self):
        return len(self.data)


