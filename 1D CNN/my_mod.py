"""Set of utility functions for ATD 2022 Challenge
    Authors: Weiyu Will Zong -- willzong@bu.edu
            Griffin Andrew Heyrich -- gheyrich@bu.edu"""

import pandas as pd
import numpy as np
import torch
import copy
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import gc


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
    tmp_stack = np.vstack((tmp_stack, last_n_rows.to_numpy()[0]))
    tmp_stack = np.vstack((tmp_stack, last_n_rows.to_numpy()[1]))
    tmp_stack=np.delete(tmp_stack, 0, axis=0)
    model.eval()
    prediction = []
    #batch_size = 1

    for i in range(n_steps):
        tmp_2d_arr = tmp_stack[-2:]
        tmp_3d_arr = tmp_2d_arr.reshape(1,tmp_2d_arr.shape[0], tmp_2d_arr.shape[1])
        preds = model(torch.tensor(tmp_3d_arr).to(device).float()).cpu().detach().numpy()
        preds = preds.reshape(20)
        #print(preds)
        tmp_stack = np.vstack((tmp_stack, preds))
        prediction.append(preds)
    

    return prediction







def generate_region_pred(region_df:pd.DataFrame, n_lags:int, epochs:int):
    x,y = getMultiDXY(region_df, n_lags=n_lags)
    y_train=y
    n_features = 20
    x_train=x.reshape((x.shape[0], x.shape[1], n_features))
    model = CNN_ForecastNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()
    train = myDataset(x_train,y_train)
    train_loader = torch.utils.data.DataLoader(train,batch_size=1,shuffle=False)
    epochs = epochs
    for epoch in range(epochs):
        print('epochs {}/{}'.format(epoch+1,epochs))
        Train(model, optimizer, train_loader, criterion)
        #Valid()
        gc.collect()
    
    res = region_df.tail(2)
    pred = prod_pred_next_n(res, model)

    return pred




def generate_full_pred(full_df:pd.DataFrame, n_epochs=200):
    name_lst = []
    for i in range(0,full_df.shape[1],20):
        name = full_df.columns[i][0]
        name_lst.append(name)

    pred_lst=[]
    for region_name in name_lst:
        region_df = full_df[region_name]
        pred = generate_region_pred(region_df, n_lags=2, epochs=n_epochs)
        pred = np.round(pred)

        col_lst=[]
        for i in range(1,21):
            col=(region_name, i)
            col_lst.append(col)
        
        cols=pd.MultiIndex.from_tuples(col_lst)
        pred_lst.append(pd.DataFrame(data=pred, columns=cols))
    
    final = pd.concat(pred_lst, axis=1)
    return final

def pre_trained_region_pred(model, region_df:pd.DataFrame, n_lags=2):
    x,y = getMultiDXY(region_df, n_lags=n_lags)
    y_train=y
    n_features = 20
    x_train=x.reshape((x.shape[0], x.shape[1], n_features))
    res = region_df.tail(2)
    pred = prod_pred_next_n(res, model)
    return pred



#----------1-D CNN Model Version 1----------
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import gc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN_ForecastNet(nn.Module):
    def __init__(self):
        super(CNN_ForecastNet,self).__init__()
        self.conv1d = nn.Conv1d(2,128,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,1)
        
    def forward(self,x):
        x = self.conv1d(x)
        x = self.relu(x)
        #print(x.shape)
        x = x.view(-1,128)
        #print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

class myDataset(Dataset):
    def __init__(self,feature,target):
        self.feature = feature
        self.target = target
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self,idx):
        item = self.feature[idx]
        label = self.target[idx]
        
        return item,label


def Train(model, optimizer, train_loader, criterion):
    train_losses = []
    running_loss = .0
    
    model.train()
    
    for idx, (inputs,labels) in enumerate(train_loader):
        inputs=inputs.to(torch.float32)
        labels=labels.to(torch.float32)
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none = True)
        preds = model(inputs.float())
        loss = criterion(preds,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss
        
    train_loss = running_loss/len(train_loader)
    train_losses.append(train_loss.cpu().detach().numpy())
    
    print(f'train_loss {train_loss}')

#index = output.cpu().data.numpy().argmax()


