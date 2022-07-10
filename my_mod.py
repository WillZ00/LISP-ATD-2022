"""Set of utility functions for ATD 2022 Challenge
    Authors: Weiyu Will Zong -- willzong@bu.edu
            Griffin Andrew Heyrich -- gheyrich@bu.edu"""

import pandas as pd
import numpy as np


def getXY(seq:pd.Series, n_lags:int):
    """Function that returns the input X and target Y for model training/validation purposes.
    Input: a time series in pandas Series format; number of lags to include"""

    x, y = [], []
    for i in range(len(seq)):
        end_idx = i + n_lags

        if end_idx > len(seq)-1:
            break
        x.append(seq[i:end_idx])
        y.append(seq[end_idx])
    
    return np.array(x), np.array(y)

def getValidation(df:pd.DataFrame):
    """Funcion that returns the validation set (forecast horizon=4)
        Input: a data frame contains the given time series
        Note: the validation set will be removed from the input dataframe"""

    validation=df.tail(6)
    df.drop(df.tail(4).index, inplace=True)
    return validation

def getMultiDXY(df: pd.DataFrame, n_lags=int):
    """Function that returns the input X and target Y for model training/validation purposes.
    Input: a time series in pandas Series format; number of lags to include"""

    x, y = [],[]
    
    tmp_stack = np.zeros(shape=(df.shape[1]))
    #print(tmp_stack.shape)
    for idx in range(len(df)):
        vector = df.iloc[idx].to_numpy()
        #series = np.reshape(series, (1,-1))
        tmp_stack = np.vstack((tmp_stack, vector))

    tmp_stack=np.delete(tmp_stack, 0, axis=0)
    #print(tmp_stack.shape)

    for i in range(len(tmp_stack)):
        end_ix = i + n_lags
        if end_ix > len(tmp_stack)-1:
            break
        seq_x, seq_y = tmp_stack[i:end_ix, :], tmp_stack[end_ix, :]
        x.append(seq_x)
        y.append(seq_y)

    return np.array(x), np.array(y)

