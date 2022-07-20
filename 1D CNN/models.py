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

@dataclass
class CnnForecaster:
    training_epochs: int = 800

    def fit(self, data: pd.DataFrame(), past_covariates=None) -> "CnnForecaster":
        full_df=data
        name_lst = []
        for i in range(0,full_df.shape[1],20):
            name = full_df.columns[i][0]
            name_lst.append(name)
        self.model_list=[]
        for region_name in name_lst:
            region_df = full_df[region_name]
            x, y_train = util.getMultiDXY(df=region_df, n_lags=2)
            n_features = 20
            x_train=x.reshape((x.shape[0], x.shape[1], n_features))
            self.model = CNN_ForecastNet().to(device)  # save it for later
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
            criterion = nn.MSELoss()
            train = util.myDataset(x_train,y_train)
            train_loader = torch.utils.data.DataLoader(train,batch_size=1,shuffle=False)
            training_epochs = 800
            epochs = training_epochs

            self.training_df = data

            for epoch in range(epochs):
                print('epochs {}/{}'.format(epoch+1,epochs))
                util.Train(self.model, optimizer, train_loader, criterion)
                gc.collect()
            
            self.model_list.append(self.model)
        return self


    def predict(self, x: pd.Index) -> pd.DataFrame:

        train_df = self.training_df
        predictions = self.generate_pred(full_df=train_df)
        predictions[predictions<0]=0
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
