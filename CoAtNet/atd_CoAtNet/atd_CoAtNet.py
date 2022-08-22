import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import os
import time
import numpy as np
import pandas as pd
from data.data_loader import atdDataset
from model.CoAtNet import CoAtNet



#ars = {"use_gpu", "batch_size", "lr", "train_epochs"}

class ATD_CoAtNet(object):
    def __init__(self, args, df:pd.DataFrame):
        self.args = args
        self.device = self._acquire_device()
        self.ATD_CoAtNet = self._build_model(df).model
        self.model = self.ATD_CoAtNet.to(self.device)

    def _build_model(self, df):
        model = CoAtNet(in_ch=1,image_size=20).float()
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

    def _get_data(self):
        data_set = atdDataset(df = self.df)
        data_loader = DataLoader(
            data_set,
            batch_size = self.args.batch_size,
            shuffle=False,
            drop_last=True
        )
        return data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self):
        pass

    def train(self):
        model =self.model
        device = self.device
        train_loader = self._get_data()
        time_now = time.time()
        #train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_losses=[]

        for epoch in range(self.args.train_epochs):
            
            for idx, (inputs, labels) in enumerate(train_loader):
                inputs=inputs.to(torch.float32)
                labels=labels.to(torch.float32)
                inputs = inputs.to(device)
                labels = labels.to(device)
                model_optim.zero_grad(set_to_none = True)
                preds = model(inputs.float())
                #preds = preds.reshape(labels.shape[0], labels.shape[1])
                print("check shapes_after", preds.shape, labels.shape)
                loss = criterion(preds,labels)
                loss.backward()
                model_optim.step()
                running_loss += loss

            train_loss = running_loss/len(train_loader)
            train_losses.append(train_loss.cpu().detach().numpy())
            print(f'train_loss {train_loss}')

        return model
