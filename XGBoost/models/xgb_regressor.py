from xgboost import XGBRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg

import pandas as pd
import numpy as np

class XGB_Res_Forecaster():
    def __init__(self, df:pd.DataFrame, args=None):
        self.args = args
        self.df = df
        self.col_names = df.columns
        #self.last_indicies = df.tail(4).index
        self.model_lst = []
    
    def train(self):
        cnt=1
        for col_name in self.df.columns:
            print(cnt)
            
            res_series = self.df[col_name]
            res_forecaster = ForecasterAutoreg(
                regressor = XGBRegressor(n_jobs = 100, 
                             tree_method = "approx"
                             ),
                lags = self.args.res_lag
                )
            res_forecaster.fit(y=res_series)
            self.model_lst.append(res_forecaster)
            cnt+=1
        return self
    
    def predict(self, indicies):
        pred_lst=[]
        for model in model_lst:
            pred = model.predict(steps =4).to_list()
            pred_lst.append(pred)
        pred_df = pd.DataFrame(data = pred_lst, index = self.col_names)
        pred_df = pred_df.set_index(indicies)
        return

        