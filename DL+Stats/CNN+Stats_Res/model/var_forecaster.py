import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from utils.tools import dotdict


class VarForecaster:

    def __init__(self, args: dotdict):
        self.args = args

    def fit(self, df: pd.DataFrame, past_covariates=None):
        self.df = df
        self.training = self._fit_processing(self.df.values)
        #print(self.training.shape)
        self.model = VAR(self.training).fit(self.args.lag)
        return self

    def predict(self, input_x:np.ndarray, n_steps = None):
        if n_steps==None:
            n_steps = self.args.predict_len
        predict = self.model.forecast(y=input_x[-self.args.lag:], steps=n_steps)
        predict_back = self._intermediate_predict_processing(predict)
        return pd.DataFrame(predict_back, columns=self.df.columns)
    
        #return predict_back
    def predict_final(self,indices):
        # if n_steps==None:
        #     n_steps = self.args.predict_len
        predict = self.model.forecast(y=self.training[-self.args.lag:], steps=self.args.predict_len)
        predict_back = self._predict_processing(predict)
        return pd.DataFrame(predict_back, columns=self.df.columns, index=indices)

    def _fit_processing(self, data:np.ndarray, flag='train'):
        if self.args.if_filter_constant:
            if flag=='train':
                self.constant_columns_idx = np.where(np.all(np.diff(data, axis=0)==0, axis=0))[0]
                self.not_constant_columns_idx = np.delete(np.arange(data.shape[1]), self.constant_columns_idx)
                self.constant_predict = data[-self.args.predict_len:, self.constant_columns_idx]
            #print(data)
            data = data[:, self.not_constant_columns_idx]
        if self.args.if_normalize:
            data, self.mean, self.std = self._normalize_data(data)
        return data




    def _intermediate_predict_processing(self, data):
        if self.args.if_normalize:
            data_back = self._verse_normalize_data(data,  self.mean, self.std)
        else:
            data_back = data

        data_back = np.round(data_back)
        data_back[data_back < 0] = 0

        if self.args.if_filter_constant:
            data_back = np.insert(data_back, self.constant_columns_idx - np.arange(len(self.constant_columns_idx)), self.constant_predict, axis=1)
            return data_back
        return data_back


    def _predict_processing(self, data):
        if self.args.if_normalize:
            data_back = self._verse_normalize_data(data,  self.mean, self.std)
        else:
            data_back = data

        data_back = np.round(data_back)
        data_back[data_back < 0] = 0

        if self.args.if_filter_constant:
            data_back = np.insert(data_back, self.constant_columns_idx - np.arange(len(self.constant_columns_idx)), self.constant_predict, axis=1)
            return data_back
        return data_back

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