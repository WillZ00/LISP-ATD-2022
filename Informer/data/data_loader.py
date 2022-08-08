import os
import numpy as np
import pandas as pd
import atd2022

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


class atdDataset(Dataset):
    def __init__(self, flag='train', size=None,
                 inverse=False, timeenc=0, freq='w', cols=None):
        # size [seq_len, label_len, pred_len]
        # info

        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        #self.features = features
        #self.target = target
        #self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols

        self.__read_data__()

    # Need to modify
    def __read_data__(self):
        df_raw = atd2022.io.read_csv()
        df_raw.insert(0, "timeStamps", df_raw.index)

        #self.scaler = StandardScaler()

        # border shape = [#train, #Vali, #Test]
        #border1s = [0, 180-self.seq_len, 180-self.seq_len]
        #border2s = [180, 215, 215]
        #border1 = border1s[self.set_type]
        #border2 = border2s[self.set_type]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.1)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        
        cols_data=df_raw.columns[self.cols:self.cols+20]
        

        df_data = df_raw[cols_data]

        #if self.scale:
        #    train_data = df_data[border1s[0]:border2s[0]]
        #    self.scaler.fit(train_data.values)
        #    data = self.scaler.transform(df_data.values)
        #else:
        #    data = df_data.values
        data = df_data.values


        df_stamp = df_raw[['timeStamps']][border1:border2]
        df_stamp["timeStamps"]=df_stamp["timeStamps"].dt.to_timestamp('W')
        df_stamp['date'] = pd.to_datetime(df_stamp.timeStamps)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        #print("begin, end", s_begin, s_end)
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        #print("r_b, r_n", r_begin, r_end)

        seq_x = self.data_x[s_begin:s_end]

        #print("dim:", seq_x.shape, )

        #print("pairs", s_begin, s_end)
        #print("y_pairs", r_begin, r_end)



        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        #print("dim:", seq_x.shape, seq_x_mark.shape)


        #print("x_mark", seq_x_mark)
        #print("y_mark", seq_y_mark)

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class atd_Pred(Dataset):
    def __init__(self, flag='train', size=None,
                 inverse=False, timeenc=0, freq='w', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        #self.features = features
        #self.target = target
        #self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
    
        #self.cols=cols
        #self.root_path = root_path
        #self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = atd2022.io.read_csv()
        df_raw.insert(0, "timeStamps", df_raw.index)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''

        border1 = len(df_raw)-30
        border2 = len(df_raw)
        
        #cols_data=df_raw.columns[1:]

        cols_data=df_raw.columns[self.cols:self.cols+20]
        df_data = df_raw[cols_data]


        data = df_data.values

        df_stamp = df_raw[['timeStamps']][border1:border2]
        df_stamp["timeStamps"]=df_stamp["timeStamps"].dt.to_timestamp('s')
        df_stamp['date'] = pd.to_datetime(df_stamp.timeStamps)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        #print("check data_x shape",self.data_x.shape)
        #print("check data_y shape",self.data_y.shape)

        #print("data_x", self.data_x)
        #print("data_y", self.data_y)
    
    def __getitem__(self, index):

        
        s_begin = index
        s_end = s_begin + self.seq_len


        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        #print("check_indices", s_begin, s_end, r_begin, r_end)


        #print("check_r_sizes", r_begin, r_end, self.label_len, self.pred_len)

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        #print("x,y", seq_x, seq_y, seq_x_mark, seq_y_mark)

        

        #print("seq", seq_x.shape)
        #print("mark",seq_x_mark.shape)

        #print("check sizes", seq_x.shape, seq_y.shape, seq_x_mark.shape, seq_y_mark.shape)

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)