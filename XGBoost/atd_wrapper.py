import atd2022
import numpy as np
import pandas as pd
from models.var_forecaster import VarForecaster
from models.xgb_regressor import XGB_Res_Forecaster

class VAR_Xgb_Forecaster:

    def __init__(self, args):
        self.args = args
    
    def fit(self, df:pd.DataFrame, past_covariates=None) -> "VAR_Xgb_Forecaster":
        self.df = df
        self.lispStats = VarForecaster(self.args)
        self.lispStats.fit(df)
        #print("stats done")
        pred_df_lst = []
        for idx in range(len(self.df)-self.args.lag):
            pred_row = self.lispStats.predict(input_x = self.df.head(self.args.lag+idx), n_steps=1)
            pred_df_lst.append(pred_row)
        pred_df = pd.concat(pred_df_lst, axis=0)
        truth_df = self.df.tail(len(self.df)-self.args.lag)
        print("stats done")
        self.res_df = truth_df - pred_df
        print(self.res_df.shape)
        self.lispML = XGB_Res_Forecaster(self.res_df, self.args)
        self.lispML.train()
        
        return self
    
    def predict(self, indices):
        stats_pred = self.lispStats.predict(indices)
        res_pred = self.lispML.predict(indices)
        return stats_pred+res_pred

