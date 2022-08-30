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
        self.lispStats_final = VarForecaster(self.args)
        self.lispStats_final.fit(df)
        tmp_df = df.head(len(df//2))
        tmp_df_np = tmp_df.values
        self.lispStats.fit(tmp_df)
        pred_df_lst = []
        for idx in range(len(df)//2):
            pred_row = self.lispStats.predict(input_x = tmp_df_np[:self.args.lag+idx], n_steps=1).set_index(self.df.index[[self.args.lag+idx]])
            #pred_row = self.lispStats.predict(input_x = self.lispStats.training[:self.args.lag+idx], n_steps=1).set_index(self.df.index[[self.args.lag+idx]])
            pred_df_lst.append(pred_row)
        self.pred_df = pd.concat(pred_df_lst, axis=0)
        #print(pred_df)
        self.truth_df = self.df.iloc[self.args.lag:]
        print("stats done")
        #print("check equal", truth_df.equals(pred_df))
        self.res_df = self.truth_df.subtract(self.pred_df)
        #print(self.res_df)
        # for i in self.res_df.columns:
        #     print((df[i] == 0).all())
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
        stats_pred = self.lispStats_final.predict_final(indicies)
        #print(stats_pred.shape)
        #print(predictions.shape)
        #print(predictions)
        final_pred = stats_pred+predictions
        final_pred[final_pred<=0]=0
        final_pred=final_pred.round()

        return final_pred

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
            #pred = np.round(pred)
            model.update_df(pred)
        # pred = model.predict()
        # print(pred)
        # print(type(pred))
        # print(pred.shape)
        

        final = model.df.tail(forecaster_horizon)
        return final