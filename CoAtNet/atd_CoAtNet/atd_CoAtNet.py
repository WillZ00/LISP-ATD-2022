import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import os
import time
import numpy as np
import pandas as pd
from data.data_loader import atdDataset, atd_Pred
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
        self.df = df.stack(level=0).sort_index(level=1)
        return self

    def update_df(self, new_row, region_name):
        data = self.df
        #last_idx=tmp.index[-1]

        #tmp.loc[last_idx+1] = new_row
        #print(new_row)

        last_idx = data.loc[(slice(None), region_name),:].index[-1][0]
        new_row_s= pd.Series(new_row, name=(last_idx+1, region_name), index=data.columns)

        self.df = data.append(new_row_s).sort_index(level=1)
        return

    # def insert_one_row(data, pred, region_name):
    #     last_idx = data.index[-1][0]
    #     new_row = pd.Series(pred, name=(last_idx+1, region_name), index=data.columns)
    #     return data.append(new_row)
    

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
            data_set = atdDataset(region_name, df = self.df)
            data_loader = DataLoader(
            data_set,
                batch_size = self.args.batch_size,
                shuffle=False,
                drop_last=True
            )
        else:
            #print("got here")
            data_set = atd_Pred(region_name, df = self.df)
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
        criterion =  nn.MSELoss()
        return criterion

    def vali(self):
        pass

    def train(self):
        model =self.model
        device = self.device
        #train_loader = self._get_data(flag="train")
        time_now = time.time()
        #train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        #train_loss=[]
        model.train()
        #running_loss= .0
        #n_step=0
        #print(self.df)
        
        name_lst = self.df.index.droplevel(0).drop_duplicates().tolist()

        for region_name in name_lst:
            train_loader = self._get_data(region_name=region_name, flag="train")
            for epoch in range(self.args.train_epochs):
                train_loss=[]
                for idx, (inputs, labels) in enumerate(train_loader):
                    inputs=inputs.to(torch.float32)
                    labels=labels.to(torch.float32)
                    inputs = inputs.unsqueeze(dim=1)
                    #labels = labels.unsqueeze(dim=1)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    #print('check input/label dim', inputs.shape, labels.shape)
                    model_optim.zero_grad(set_to_none = True)
                    preds = model(inputs.float())
                    #preds = preds.reshape(labels.shape[0], labels.shape[1])
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

    def predict(self, region_name:str):
        model =self.model
        device = self.device
        pred_loader = self._get_data(region_name, flag="pred")
        time_now = time.time()

        model.eval()
        preds = []
        
        #print(len(pred_loader))
        for idx, (inputs, mean, std) in enumerate(pred_loader):
            #print("got here")
            #print(inputs)
            #print(inputs.shape)
            inputs = inputs.unsqueeze(dim=1)
            pred = model(torch.tensor(inputs).to(device).float()).cpu().detach().numpy()
            #pred = (pred*std)+mean
            #print(pred)
            #print(type(pred))
            #print("checkstd",type(std))
            #print("checkmean",type(mean))
            std = std.cpu().detach().numpy()
            std[std==0]=1
            pred = (pred*std)+mean.cpu().detach().numpy()
            preds.append(pred)
        preds=np.array(preds)
        #print(preds)
        #preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        return np.squeeze(preds)



