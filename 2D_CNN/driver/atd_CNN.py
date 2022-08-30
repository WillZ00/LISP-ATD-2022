import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import os
import time
import numpy as np
import pandas as pd

from data.data_loader import atd_dataset, atd_Pred
from model.CNN_2D import CNN_ForecastNet

class ATD_CNN(object):
    def __init__(self, args, df:pd.DataFrame):
        self.args = args
        self.device = self._acquire_device()
        self.CNN_ForecastNet = self._build_model(df).model
        self.model = self.CNN_ForecastNet.to(self.device)

    
    def _build_model(self, df):
        self.dim = self.args.dim
        self.history_len=self.args.history_len
        model = CNN_ForecastNet()
        self.model = model
        self.df = df
        return self
    
    def update_df(self, new_row):
        #tmp = self.df.drop(["timeStamps"], axis=1)
        tmp = self.df
        last_idx=tmp.index[-1]
        tmp.loc[last_idx+1] = new_row
        
        #tmp.insert(0, "timeStamps", tmp.index)
        self.df = tmp
        return


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
    

    def _get_data(self, flag:str):
        history_len = self.args.history_len

        if flag=="train":
            data_set = atd_dataset(df = self.df,history_len= history_len)
            data_loader = DataLoader(
            data_set,
                batch_size = self.args.batch_size,
                shuffle=False,
                drop_last=True
            )
        else:
            #print("got here")
            data_set = atd_Pred(df = self.df, history_len = history_len)
            #print(data_set)
            data_loader = DataLoader(
                data_set,
                batch_size = 1,
                shuffle=False,
                drop_last=True
            )
        return data_loader
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.SmoothL1Loss()
        return criterion
    
    def train(self):
        model =self.model
        device = self.device
        time_now = time.time()
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_loader = self._get_data(flag="train")
        model.train()
        for epoch in range(self.args.train_epochs):
            train_loss=[]
            for idx, (inputs, labels) in enumerate(train_loader):
                inputs=inputs.to(torch.float32)
                labels=labels.to(torch.float32)
               # print('check input/label dim', inputs.shape, labels.shape)
                inputs = inputs.unsqueeze(dim=1)
                #labels = labels.unsqueeze(dim=1)
                inputs = inputs.to(device)
                labels = labels.to(device)
                #inputs = inputs.reshape(inputs.shape[0], inputs.shape[2], inputs.shape[1])
                #print('check input/label dim', inputs.shape, labels.shape)
                model_optim.zero_grad(set_to_none = True)
                preds = model(inputs.float())
                preds = preds.squeeze(dim=1)
                #preds = preds.reshape(labels.shape[0], labels.shape[1], labels.shape[2])
                #print("check shapes_after", preds.shape, labels.shape)
                loss = criterion(preds,labels)
                #print(type(train_loss))
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()
                #running_loss += loss
            
            train_loss = np.average(train_loss)
            #train_losses.append(train_loss.cpu().detach().numpy())
            print(f'train_loss {train_loss}')
        return self

    def predict(self):
        model =self.model
        device = self.device
        pred_loader = self._get_data(flag="pred")
        time_now = time.time()

        model.eval()
        preds = []
        
        #print(len(pred_loader))
        for idx, inputs in enumerate(pred_loader):
            inputs = inputs.unsqueeze(dim=1)
            pred = model(torch.tensor(inputs).to(device).float()).cpu().detach().numpy()

            preds.append(pred)
        preds=np.array(preds)
        #print(preds)
        #preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        return np.squeeze(preds)