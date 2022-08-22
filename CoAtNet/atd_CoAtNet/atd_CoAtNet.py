import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import os
import time
import numpy as np
import pandas as pd


class ATD_CoAtNet(object):
    def __init__(self, args, df:pd.DataFrame):
        self.args = args
        self.device = self._acquire_device()
        self.ATD_CoAtNet = self._build_model(df).model
        self.model = self.ATD_CoAtNet.to(self.device)

    def _build_model(self, df):
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda')
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass