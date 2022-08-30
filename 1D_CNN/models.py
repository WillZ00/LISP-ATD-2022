from dataclasses import dataclass
from typing import Optional
import my_mod as util
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import gc
import numpy as np
from my_mod import CNN_ForecastNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#@dataclass
class CnnForecaster:

    def __init__(self, args):
        self.training_epochs = args.epochs
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.d_model = args.d_model


    def fit(self, data: pd.DataFrame(), past_covariates=None) -> "CnnForecaster":
        training_epochs = self.training_epochs
        batch_size = self.batch_size
        d_model = self.d_model
        full_df=data
        name_lst = []
        for i in range(0,full_df.shape[1],20):
            name = full_df.columns[i][0]
            name_lst.append(name)
        self.model_list=[]
        current_cnt = 0
        for region_name in name_lst:
            print(current_cnt)
            region_df = full_df[region_name]
            x, y_train = util.getMultiDXY(df=region_df, n_lags=2)
            n_features = 20
            x_train=x.reshape((x.shape[0], x.shape[1], n_features))

            #print(x_train.shape, y_train.shape)
            self.model = CNN_ForecastNet(dim=d_model).to(device)  # save it for later
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            criterion = nn.MSELoss()
            train = util.myDataset(x_train,y_train)

            train_loader = torch.utils.data.DataLoader(train,batch_size=batch_size,shuffle=False)
            #training_epochs = 200
            epochs = training_epochs

            self.training_df = data

            for epoch in range(epochs):
                print('epochs {}/{}'.format(epoch+1,epochs))
                util.Train(self.model, optimizer, train_loader, criterion)
                gc.collect()
            
            self.model_list.append(self.model)
            current_cnt+=1
        return self


    def predict(self, indicies) -> pd.DataFrame:
        train_df = self.training_df
        predictions = self.generate_pred(full_df=train_df)
        predictions[predictions<0]=0
        predictions = predictions.set_index(indicies)

        return predictions


    def generate_pred(self, full_df:pd.DataFrame):
        model_list = self.model_list

        name_lst = []
        for i in range(0,full_df.shape[1],20):
            name = full_df.columns[i][0]
            name_lst.append(name)
        current_iter=0
        pred_lst=[]
        for region_name in name_lst:
            current_model = model_list[current_iter]
            region_df = full_df[region_name]
            pred = util.pre_trained_region_pred(model=current_model, region_df=region_df, n_lags=2)
            pred = np.round(pred)

            col_lst=[]
            for i in range(1,21):
                col=(region_name, i)
                col_lst.append(col)
        
            cols=pd.MultiIndex.from_tuples(col_lst)
            pred_lst.append(pd.DataFrame(data=pred, columns=cols))
    
        final = pd.concat(pred_lst, axis=1)
        return final
