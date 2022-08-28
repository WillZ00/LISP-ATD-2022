import atd2022
import torch
import numpy as np
import pandas as pd
from driver.atd_CNN import ATD_CNN
from utils.tools import dotdict
from model.var_forecaster import VarForecaster

class CNN_VAR_Forecaster():

    def __init__(self, args):
        self.args = args

    def fit(self, df:pd.DataFrame, past_covariates=None) -> "CNN_VAR_Forecaster":
        self.df=df
        self.lispStats = VarForecaster(self.args)
        self.lispStats.fit(df)
        pred_df_lst = []
        for idx in range(len(self.df)-self.args.lag):
            pred_row = self.lispStats.predict(input_x = self.df.head(self.args.lag+idx), n_steps=1)
            pred_df_lst.append(pred_row)
        pred_df = pd.concat(pred_df_lst, axis=0)
        truth_df = self.df.tail(len(self.df)-self.args.lag)
        print("stats done")
        self.res_df = truth_df - pred_df
        print(res_df)
        # pred_df = self.lispStats.predict(input_x = self.df.head(self.args.lag), steps=len(self.df)-self.args.lag)
        # pred_df = pd.concate([self.df.head(self.args.lag), pred_df], axis=0)
        # res_df = self.df-pred_df
        lispDL = ATD_CNN(self.args, self.res_df)
        lispDL.train()
        self.model = lispDL
    
        return self


    def predict(self, indicies):
       # print(indicies)
        predictions = self.generate_pred(indicies)
        #print(predictions.shape)
        #time_idxs=predictions.index
        #predictions[predictions<=0]=0
        predictions = predictions.to_numpy()
        predictions = pd.DataFrame(data=predictions, index = indicies, columns=self.df.columns)
        stats_pred = self.lispStats.predict_final(indicies)
        #print(stats_pred.shape)
        #print(predictions.shape)
        final_pred = stats_pred+predictions
        final_pred[final_pred<=0]=0
        final_pred=final_pred.round()

        return predictions

    def generate_pred(self, indicies):
        forecaster_horizon = len(indicies)

        model = self.model
        #if "timeStamps" in self.df.columns:
        #    self.df = self.df.drop(["timeStamps"], axis=1)
        #print("len of indicies is ", len(indicies))
        #print(forecaster_horizon)

        for j in range(forecaster_horizon):
            #print("got here")
            pred = model.predict()
            pred = np.round(pred)
            model.update_df(pred)
        # pred = model.predict()
        # print(pred)
        # print(type(pred))
        # print(pred.shape)
        

        final = model.df.tail(forecaster_horizon)
        return final