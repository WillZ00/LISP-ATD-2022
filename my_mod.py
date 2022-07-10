"""Set of utility functions for ATD 2022 Challenge
    Authors: Weiyu Will Zong -- willzong@bu.edu
            Griffin Andrew Heyrich -- gheyrich@bu.edu"""

import pandas as pd
import numpy as np
import torch
import copy


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

def get_testXY(region_df:pd.DataFrame, n_steps:int, n_lags:int):
    return getMultiDXY(region_df.tail(n_steps+2), n_lags=n_lags)

def pred_next_n(x, model, n_steps:int):
    """Note: n_steps must be greater than or equals to 4
        Function that predicts the next N time steps using a given model"""

    x_test = copy.deepcopy(x)
    
    model.eval()
    prediction = []
    batch_size = 1  
    iterations = n_steps-2
    for i in range(iterations):
        preds = model(torch.tensor(x_test[batch_size*i:batch_size*(i+1)]).float()).detach().numpy()
        preds = preds.reshape(1,20)
        x_test[(i+1)][1]=preds
        x_test[(i+2)][0]=preds
        prediction.append(preds)

    third_pred = model(torch.tensor(x_test[-1]).float()).detach().numpy().reshape(1,20)
    x_test[-1][1]=third_pred
    prediction.append(third_pred)

    fourth_pred = model(torch.tensor(x_test[-1]).float()).detach().numpy().reshape(1,20)
    prediction.append(fourth_pred)
    
    return prediction


def metrics_mse(pred, truth):
    from sklearn.metrics import mean_squared_error
    for i in range(len(pred)):
        y_true = truth[i]
        y_pred = pred[i].reshape(20)
        print(mean_squared_error(y_true, y_pred))







#---------util functions for full-scale production below---------

def prod_pred_next_n(last_n_rows, model, n_steps=4):

    tmp_stack = np.zeros(shape=(20))
    print("1")
    tmp_stack = np.vstack((tmp_stack, last_n_rows.to_numpy()[0]))
    print("2")
    tmp_stack = np.vstack((tmp_stack, last_n_rows.to_numpy()[1]))
    print("3")
    tmp_stack=np.delete(tmp_stack, 0, axis=0)
    print("4")
    print(tmp_stack.shape)
    model.eval()
    prediction = []
    #batch_size = 1

    for i in range(n_steps):
        preds = model(torch.tensor(tmp_stack[i:(i+1)]).float()).detach().numpy()
        preds = preds.reshape(20)
        print(preds)
        tmp_stack = np.vstack((tmp_stack, preds))
        prediction.append(preds)

    return prediction



    






def generate_full_pred():


    return
