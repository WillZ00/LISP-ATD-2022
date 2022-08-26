import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from utils.tools import dotdict


class VarForecaster:

    def __init__(self, args: dotdict):
        self.args = args

    def fit(self, df: pd.DataFrame, past_covariates=None):
        self.df = df
        self.training = self._fit_processing(self.df)
        #print(self.training.shape)
        self.model = VAR(self.training).fit(self.args.lag)
        return self

    def predict(self, indicies):
        predict = self.model.forecast(y=self.training[-self.args.lag:], steps=self.args.predict_len)
        predict_back = self._predict_processing(predict)
        return predict_back[self.df.columns].set_index(indicies)

    def _fit_processing(self, data):
        if self.args.if_filter_constant:
            self.constant_columns = data.diff().loc[:, (data.diff().iloc[1:] == 0).all()].columns
            data = data.drop(self.constant_columns, axis=1)
            self.not_constant_columns = data.columns
        if self.args.if_normalize:
            data_proc, self.mean, self.std = self._normalize_data(data)
        else:
            data_proc = data
        return data_proc

    def _predict_processing(self, data):
        if self.args.if_normalize:
            data_back = self._verse_normalize_data(data,  self.mean, self.std)
        else:
            data_back = data

        data_back = np.round(data_back)
        data_back[data_back < 0] = 0

        if self.args.if_filter_constant:
            data_back_df = pd.DataFrame(data_back, columns=self.not_constant_columns)
            data_back_df[self.constant_columns] = self.df.iloc[-self.args.predict_len:][self.constant_columns].values
            return data_back_df
        return pd.DataFrame(data_back, columns=self.df.columns)

    def _normalize_data(seft, data):
        data = np.array(data)
        # data_mean = np.average(data, axis=0)
        data_mean = 0
        data_std = np.std(data, axis=0)
        data_std[data_std == 0] = 1
        data_normalized = (data - data_mean) / data_std
        return data_normalized, data_mean, data_std

    def _verse_normalize_data(self, data, data_mean, data_std):
        return data * data_std + data_mean