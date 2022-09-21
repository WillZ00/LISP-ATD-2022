import atd2022
import torch
import numpy as np
import pandas as pd
from driver.atd_ET_CNN import ATD_ET_CNN
from utils.tools import dotdict

class ET_CNN_Forecaster():

    def __init__(self, args):
        self.args = args

    def fit(self, df:pd.DataFrame, past_covariates=None) -> "CNN_Transformer_Forecaster":
        self.df=df
        self.training = pd.DataFrame(self._fit_processing(df), columns=df.columns, index=df.index)
        # add position encoding
        exp = ATD_ET_CNN(self.args, self.training)
        exp.train()
        self.model = exp
        return self


    def predict(self, indicies):
        predictions = self.generate_pred(indicies)
        predictions_df = self._predict_processing(predictions.values)
        return predictions_df.set_index(indicies)

    def generate_pred(self, indicies):
        forecaster_horizon = len(indicies)

        model = self.model
        if "timeStamps" in self.df.columns:
            self.df = self.df.drop(["timeStamps"], axis=1)


        for j in range(int(forecaster_horizon/self.args.predict_len)):
            pred = model.predict()
            pred = np.round(pred)
            pred = pred.reshape(self.args.predict_len, 5200)
            model.update_df(pred)
        # pred = model.predict()
        # print(pred)
        # print(type(pred))
        # print(pred.shape)
        

        final = model.df.tail(forecaster_horizon)
        return final

    def _fit_processing(self, data):
        if self.args.if_filter_constant:
            self.constant_columns = data.diff().loc[:, (data.diff().iloc[1:] == 0).all()].columns
            # data = data.drop(self.constant_columns, axis=1)
            self.not_constant_columns = data.columns
        if self.args.if_normalize:
            data_proc, self.mean, self.std = self._normalize_data(data)
        else:
            data_proc = np.array(data)
        return data_proc

    def _predict_processing(self, data):
        if self.args.if_normalize:
            data_back = self._verse_normalize_data(data,  self.mean, self.std)
        else:
            data_back = data

        data_back = np.round(data_back)
        data_back[data_back < 0] = 0

        if self.args.if_filter_constant:
            data_back_df = pd.DataFrame(data_back, columns=self.df.columns)
            data_back_df[self.constant_columns] = self.df.iloc[-4:][self.constant_columns].values
            return data_back_df
        return pd.DataFrame(data_back, columns=self.df.columns)

    def _normalize_data(seft, data):
        data = np.array(data)
        data_mean = np.average(data, axis=0)
        # data_mean = 0
        data_std = np.std(data, axis=0)
        data_std[data_std == 0] = 1
        data_normalized = (data - data_mean) / data_std
        return data_normalized, data_mean, data_std

    def _verse_normalize_data(self, data, data_mean, data_std):
        return data * data_std + data_mean

