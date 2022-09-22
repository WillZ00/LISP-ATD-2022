import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg
from utils.tools import dotdict


class VarForecaster:

    def __init__(self, args: dotdict):
        self.args = args

    def fit(self, df: pd.DataFrame, past_covariates=None):
        self.df = df
        self.training = self._fit_processing(self.df.values, flag='train')
        if self.args.seperate_train:
            df_columns = self.df.columns
            if self.args.if_filter_constant:
                df_columns = df_columns[self.not_constant_columns_idx]
            self.training_df = pd.DataFrame(self.training, columns=df_columns)
            self.model_dict = {}
            for region_code, region_training in self.training_df.groupby(axis=1,level=0):
                if region_training.shape[1] > 1:
                    self.model_dict[region_code] = VAR(region_training.values).fit(self.args.lag)
                else:
                    self.model_dict[region_code] = AutoReg(region_training.values, lags=self.args.lag).fit()
        else:
            self.model = VAR(self.training).fit(self.args.lag)
        return self

    def predict(self, indicies):
        if self.args.seperate_train:
            predict = np.empty((self.args.predict_len, 0))
            for region_code, region_training in self.training_df.iloc[-self.args.lag:].groupby(axis=1,level=0):
                if region_training.shape[1] > 1:
                    region_predict = self.model_dict[region_code].forecast(y=region_training.values, steps=self.args.predict_len)
                else:
                    region_predict = self.model_dict[region_code].forecast(steps=self.args.predict_len).reshape((self.args.predict_len, 1))
                predict = np.hstack([predict, region_predict])
        else:
            predict = self.model.forecast(y=self.training[-self.args.lag:], steps=self.args.predict_len)
        predict_back = self._predict_processing(predict)
        return pd.DataFrame(predict_back, columns= self.df.columns, index=indicies)

    def _fit_processing(self, data:np.ndarray, flag='train'):
        if self.args.if_filter_constant:
            if flag == 'train':
                self.constant_columns_idx = np.where(np.all(np.diff(data, axis=0) == 0, axis=0))[0]
                self.not_constant_columns_idx = np.delete(np.arange(data.shape[1]), self.constant_columns_idx)
                self.constant_predict = data[-self.args.predict_len:, self.constant_columns_idx]
            data = data[:, self.not_constant_columns_idx]
        if self.args.if_normalize:
            data, self.mean, self.std = self._normalize_data(data)
        return data

    def _predict_processing(self, data):
        if self.args.if_normalize:
            data_back = self._verse_normalize_data(data,  self.mean, self.std)
        else:
            data_back = data

        if self.args.if_round:
            data_back = np.round(data_back)
            data_back[data_back < 0] = 0

        if self.args.if_filter_constant:
            data_back = np.insert(data_back, self.constant_columns_idx - np.arange(len(self.constant_columns_idx)), self.constant_predict, axis=1)

        mean_back_up = np.vstack([self.df.values.mean(axis=0)] * self.args.predict_len)
        error_predict = (data_back > self.args.error_threshold * mean_back_up)
        data_back[error_predict] = mean_back_up[error_predict]
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
