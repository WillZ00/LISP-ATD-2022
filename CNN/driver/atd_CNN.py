import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import os
import time
import numpy as np
import pandas as pd

from data.data_loader import atd_dataset
from model.CNN_1D import CNN_ForecastNet

class ATD_CNN(object):
    def __init__(self, args, df:pd.DataFrame):
        self.args = args
        self.device = self._acquire_device()
        self.ATD_CNN = self._build_model(df)
        self.model = self.ATD_CNN.to(self.device)

    
    def _build_model(self, df):
        self.dim = self.args.dim
        model = CNN_ForecastNet(dim = self.dim)
        self.model=model
        self.df = df
        return self


    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda')
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        
        self.device = device
        return device
    
    def _get_data(self, region_name:str, flag:str):
    if flag=="train":
        data_set = atd_dataset(df = self.df)
        data_loader = DataLoader(
        data_set,
            batch_size = self.args.batch_size,
            shuffle=False,
            drop_last=True
        )
    else:
        #print("got here")
        data_set = atd_Pred(df = self.df)
        #print(data_set)
        data_loader = DataLoader(
            data_set,
            batch_size = 1,
            shuffle=False,
            drop_last=True
        )
    return data_loader