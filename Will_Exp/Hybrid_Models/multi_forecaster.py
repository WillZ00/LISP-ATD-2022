import pandas as pd
import numpy as np
from utils.tools import dotdict
from var_forecaster import VarForecaster
from atd2022.forecasters import PredictMeanForecaster, PredictLastForecaster, ExponentiallyWeightedMovingAverage
#from CNN_Transformer_Forecaster_Wrapper import CNN_Transformer_Forecaster
from Event_Time_CNN.ET_CNN_Wrapper import ET_CNN_Forecaster


class MultiForecaster:

    def __init__(self, args: dotdict):
        self.df = None
        self.args = args
        self.model_name_dict = {
            'var':VarForecaster, 
            'ET_CNN': ET_CNN_Forecaster, 
            'pmf':PredictMeanForecaster,
            'plf':PredictLastForecaster, 
            'EWMA':ExponentiallyWeightedMovingAverage}
        self.model_list = []

    def fit(self, df: pd.DataFrame, past_covariates=None):
        self.df = df
        for i, model_name in enumerate(self.args.model_list):
            if self.args.get(model_name):
                self.model_list.append(self.model_name_dict[model_name](self.args[model_name]).fit(self.df))
            else:
                self.model_list.append(self.model_name_dict[model_name]().fit(self.df))
        return self

    def predict(self, indicies):
        predict_sum = pd.DataFrame(0, columns=self.df.columns, index=indicies)
        for model in self.model_list:
            predict = model.predict(indicies)
            predict_sum += predict
        return (predict_sum/len(self.model_list)).round()