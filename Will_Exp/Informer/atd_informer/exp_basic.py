import os
import torch
import numpy as np
import pandas as pd

class Exp_Basic(object):
    def __init__(self, args, df:pd.DataFrame):
        self.args = args
        self.device = self._acquire_device()
        self.ATD_Informer_model = self._build_model(df).model
        self.model = self.ATD_Informer_model.to(self.device)

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
    