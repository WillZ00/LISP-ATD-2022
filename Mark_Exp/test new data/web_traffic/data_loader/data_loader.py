import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gc
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainDataset(Dataset):
    def __init__(self, df:pd.DataFrame, section_levels, history_len, predict_len, device):
        self.df = df
        self.section_levels = section_levels
        self.history_len = history_len
        self.predict_len = predict_len
        self.device = device
        self.data = torch.Tensor(self.df.values).to(device)
        self.time_length = len(df)
        self.__read_data__()


    def __len__(self):
        return self.time_length-(self.history_len+self.predict_len-1)


    def __read_data__(self):
        self.original_columns = self.df.columns
        self.data_list = []
        self.column_indexes_list = []
        for i in self.section_levels:
            temp_data = self.df.sort_index(axis=1, level=i)
            index_value_counts = pd.Series(temp_data.columns.get_level_values(i)).value_counts().sort_index()
            insert_locations = index_value_counts.cumsum()
            max_len = index_value_counts.max()
            insert_numbers = max_len - index_value_counts
            temp_values = temp_data.values
            ones_zeros = np.ones(len(self.original_columns))
            for idx, loc in insert_locations.sort_index(ascending=False).iteritems():
                temp_values = np.insert(temp_values, loc, np.zeros((insert_numbers[idx],1)), axis=1)
                ones_zeros = np.insert(ones_zeros, loc, np.zeros(insert_numbers[idx]))
            column_indexes = np.arange(temp_values.shape[1])[ones_zeros==1]
            column_indexes = pd.Series(column_indexes, index=temp_data.columns)[self.df.columns].values
            temp_values = temp_values.reshape(self.time_length, len(index_value_counts), max_len)
            self.data_list.append(torch.Tensor(temp_values).to(self.device))
            self.column_indexes_list.append(column_indexes)


    def __getitem__(self,idx):
        train_xs = [data[idx : idx+self.history_len].permute(2,0,1) for data in self.data_list]
        train_y = self.data[idx+self.history_len : idx+self.history_len+self.predict_len]
        return train_xs, train_y


class PredDataset(Dataset):
    def __init__(self, df:pd.DataFrame, section_levels, history_len, device):
        self.df=df
        self.section_levels = section_levels
        self.history_len = history_len
        self.device = device
        self.time_length = len(df)
        # self.data = self.df.values.to(torch.float32).to(device)
        self.__read_data__()

    def __len__(self):
        return self.time_length-(self.history_len-1)

    def __read_data__(self):
        self.original_columns = self.df.columns
        self.data_list = []
        self.column_indexes_list = []
        for i in self.section_levels:
            temp_data = self.df.sort_index(axis=1, level=i)
            index_value_counts = pd.Series(temp_data.columns.get_level_values(i)).value_counts().sort_index()
            insert_locations = index_value_counts.cumsum()
            max_len = index_value_counts.max()
            insert_numbers = max_len - index_value_counts
            temp_values = temp_data.values
            ones_zeros = np.ones(len(self.original_columns))
            for idx, loc in insert_locations.sort_index(ascending=False).iteritems():
                temp_values = np.insert(temp_values, loc, np.zeros((insert_numbers[idx],1)), axis=1)
                ones_zeros = np.insert(ones_zeros, loc, np.zeros(insert_numbers[idx]))
            column_indexes = np.arange(temp_values.shape[1])[ones_zeros==1]
            column_indexes = pd.Series(column_indexes, index=temp_data.columns)[self.df.columns].values
            temp_values = temp_values.reshape(self.time_length, len(index_value_counts), max_len)
            self.data_list.append(torch.Tensor(temp_values).to(self.device))
            self.column_indexes_list.append(column_indexes)

    def __getitem__(self, idx):
        pred_xs = [data[idx : idx+self.history_len].permute(2,0,1) for data in self.data_list]
        return pred_xs