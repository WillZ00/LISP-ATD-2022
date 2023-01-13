import torch
import numpy as np
import pandas as pd
from driver.driver import SectionCNNDriver
from utils.tools import dotdict

class SectionCNNForecaster():

    def __init__(self, args: dotdict):
        self.args = args

    def fit(self, df:pd.DataFrame, past_covariates=None):
        self.df=df
        self.training = self._train_processing(df)
        self.model = SectionCNNDriver(self.args, self.training)
        self.model.train()
        return self

    def predict(self, indicies, df=None):
        if not df:
            df = self.training.iloc[-self.args.history_len:]
        else:
            df = self._train_processing(df)
        predictions = self.model.predict(df, indicies)
        if len(predictions)==1:
            predictions_df = self._predict_processing(predictions[0])
            return predictions_df
        else:
            predictions_df_list = [self._predict_processing(pred) for pred in predictions]
            return predictions_df_list


    def _train_processing(self, data:pd.DataFrame):
        if self.args.if_filter_constant:
            self.constant_columns = data.diff().loc[:, (data.diff().iloc[1:] == 0).all()].columns
            # data = data.drop(self.constant_columns, axis=1)
            # self.not_constant_columns = data.columns
        if self.args.if_normalize:
            data_proc, self.mean, self.std = self._normalize_data(data)
        else:
            data_proc = np.array(data)
        return pd.DataFrame(data_proc, columns=data.columns, index=data.index)

    def _pred_x_processing(self, data:pd.DataFrame):
        data = data[self.df.columns]
        if self.args.if_filter_constant:
            data = data.drop(self.constant_columns, axis=1)
        if self.args.if_normalize:
            data_proc = (data.values - self.mean) / self.std
        else:
            data_proc = data.values
        return pd.DataFrame(data_proc, index=data.index, columns=data.columns)

    def _predict_processing(self, df):
        data = df.values
        if self.args.if_normalize:
            data_back = self._verse_normalize_data(data,  self.mean, self.std)
        else:
            data_back = data

        data_back = np.round(data_back)
        if self.args.if_filter_negtive:
            data_back[data_back < 0] = 0

        data_back_df = pd.DataFrame(data_back, columns=df.columns, index=df.index)
        if self.args.if_filter_constant:
            data_back_df[self.constant_columns] = self.df.iloc[-len(data_back_df):][self.constant_columns].values
        return data_back_df

    def _normalize_data(seft, data):
        data = np.array(data)
        data_mean = np.average(data, axis=0)
        data_std = np.std(data, axis=0)
        data_std[data_std == 0] = 1
        data_normalized = (data - data_mean) / data_std
        return data_normalized, data_mean, data_std

    def _verse_normalize_data(self, data, data_mean, data_std):
        return data * data_std + data_mean

