import atd2022
import torch
import numpy as np
import pandas as pd
from driver.atd_CNN_Transformer import ATD_CNN_Transformer
from utils.tools import dotdict

class CNN_Transformer_Forecaster():

    def __init__(self, args):
        self.args = args

    def fit(self, df:pd.DataFrame, past_covariates=None) -> "CNN_Transformer_Forecaster":
        self.df=df
        exp = ATD_CNN_Transformer(self.args, df)
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
        if "timeStamps" in self.df.columns:
            self.df = self.df.drop(["timeStamps"], axis=1)


        for j in range(int(forecaster_horizon/self.args.predict_len)):
            pred = model.predict()
            pred = np.round(pred)
            model.update_df(pred)
        # pred = model.predict()
        # print(pred)
        # print(type(pred))
        # print(pred.shape)
        

        final = model.df.tail(forecaster_horizon)
        return final

