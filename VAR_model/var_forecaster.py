import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from utils.tools import dotdict


class VarForecaster:

    def __init__(self, args: dotdict):
        self.args = args

    def fit(self, df: pd.DataFrame, past_covariates=None):
        self.df = df
        self.model = VAR(self.df).fit(self.args.lag)
        return self

    def predict(self, indicies):
        predict = self.model.forecast(y=self.df.values[-self.args.lag:], steps=self.args.predict_len)
        predict = np.round(predict)
        predict[predict < 0] = 0
        return pd.DataFrame(predict, columns=self.df.columns, index=indicies)