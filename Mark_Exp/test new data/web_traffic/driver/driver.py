import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import time
import numpy as np
import pandas as pd

from data_loader.data_loader import TrainDataset, PredDataset
from model.finalmodel import CNN_Transformer_Net

torch.manual_seed(123)

class SectionCNNDriver(object):
    def __init__(self, args, df:pd.DataFrame):
        self.args = args
        self.df = df
        self.data_wedth = df.shape[1]
        self.device = self._acquire_device()
        self.model = CNN_Transformer_Net(args.section_structure, self.data_wedth, history_len=args.history_len, predict_len=args.predict_len)
        self.model = self.model.to(self.device)

    
    # def update_df(self, new_rows):
    #     #tmp = self.df.drop(["timeStamps"], axis=1)
    #     tmp = self.df
    #     last_idx=tmp.index[-len(new_rows):]
    #     new_df = pd.DataFrame(new_rows, index=last_idx+self.args.predict_len, columns=tmp.columns)
    #     tmp = pd.concat([tmp, new_df])
        
    #     #tmp.insert(0, "timeStamps", tmp.index)
    #     self.df = tmp
    #     return


    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda')
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    

    def _get_data(self, df, flag:str):
        history_len = self.args.history_len
        predict_len = self.args.predict_len
        section_levels = self.args.section_levels
        if flag=="train":
            data_set = TrainDataset(df, section_levels, history_len, predict_len, device=self.device)
            self.valid_column_indexes_list = data_set.column_indexes_list
            data_loader = DataLoader(data_set, batch_size = self.args.batch_size, shuffle=False,drop_last=False)
        else:
            data_set = PredDataset(df, section_levels, history_len, device=self.device)
            data_loader = DataLoader(data_set, batch_size = 1,shuffle=False,drop_last=False)
        return data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.L1Loss()
        # criterion =  nn.MSELoss()
        return criterion
    
    def train(self):
        model =self.model
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = CosineAnnealingWarmRestarts(model_optim, T_0=20, T_mult=2)

        train_loader = self._get_data(self.df, flag="train")
        model.train()
        for epoch in range(self.args.train_epochs):
            # start_time = time.time()
            train_loss=[]
            for idx, (inputs, labels) in enumerate(train_loader):
                model_optim.zero_grad(set_to_none = True)
                preds = model(inputs, self.valid_column_indexes_list)
                loss = criterion(preds,labels)
                loss.backward()
                model_optim.step()
                if self.args.if_scheduler:
                    scheduler.step()
                train_loss.append(loss.item())
            train_loss = np.average(train_loss)
            # print(f'epoch {epoch} time: {time.time() - start_time}')
            print(f'epoch {epoch} train_loss: {train_loss}')
        return self

    def predict(self, df, outcome_index):
        predict_len = self.args.predict_len
        model = self.model
        device = self.device
        pred_loader = self._get_data(df, flag="pred")
        # time_now = time.time()

        model.eval()
        preds = []
        for idx, inputs in enumerate(pred_loader):
            pred = model(inputs, self.valid_column_indexes_list).cpu().detach().numpy()
            pred = pd.DataFrame(pred, index=outcome_index[idx:idx+predict_len], columns=df.columns)
            preds.append(pred)
        return preds