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
        data = df.valuess

