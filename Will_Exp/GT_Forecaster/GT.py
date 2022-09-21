import pandas as pd
import numpy as np

class GTForecaster:
    def __init__(self, true_df):
        self.true_df = true_df


    def fit(self, df: pd.DataFrame, past_covariates=None):
        self.true_pred = self.true_df.iloc[len(df): len(df)+4]

        return self
        
    def predict(self, indicies):
        return self.true_pred