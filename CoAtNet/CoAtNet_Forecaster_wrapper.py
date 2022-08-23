import atd2022
import torch
import numpy as np
import pandas as pd
from atd_CoAtNet.atd_CoAtNet import ATD_CoAtNet
from utils.tools import dotdict


class CoAtNetForecaster():

    def __init__(self, args):
        self.args = args
    
    def fit(self, df:pd.DataFrame, past_covariates=None) -> "CoAtNetForecaster":
        self.df=df
        exp = ATD_CoAtNet(self.args, df)
        exp.train()
        self.model = exp
        return self


    def predict(self, indicies):
        predictions = self.generate_pred(indicies)
        time_idxs=predictions.index
        predictions[predictions<=0]=0
        predictions = predictions.to_numpy()
        predictions = pd.DataFrame(data=predictions, index = time_idxs, columns=self.df.columns)

        return predictions



    def generate_pred(self, indicies):
        forecaster_horizon = len(indicies)
        model = self.model

        name_lst = []
        col_lst=[]
        for i in range(0,self.df.shape[1],20):
            name = self.df.columns[i][0]
            name_lst.append(name)
        #print(name_lst)
        for name in name_lst:
            for i in range(forecaster_horizon):
                current_pred = model.predict(name)
                current_pred = np.round(current_pred)
                #cur_pred_lst.append(current_pred)
                model.update_df(current_pred, name)

        pred_lst=pd.DataFrame()
        for name in name_lst:
            # print(model.df.loc[(slice(None),name),:].tail(5))
            pred_lst = pd.concat([pred_lst,model.df.loc[(slice(None),name),:].unstack(level=1).swaplevel(axis=1).tail(forecaster_horizon)], axis=1)
        model.df = model.df.unstack().iloc[:-4].stack(level=1).sort_index(level=1)
        return pred_lst

#data.loc[(slice(None),region_name),:].unstack(level=1).swaplevel(axis=1)