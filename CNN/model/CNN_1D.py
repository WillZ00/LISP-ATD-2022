import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import gc


class CNN_ForecastNet(nn.Module):
    def __init__(self, dim=512, history_len=52):
        super(CNN_ForecastNet,self).__init__()
        self.dim = dim
        self.history_len=history_len
        self.conv1d = nn.Conv1d(history_len , self.dim,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.dim,128)
        self.fc2 = nn.Linear(128,1)
        
    def forward(self,x):
        x = self.conv1d(x)
        x = self.relu(x)
        #print(x.shape)
        x = x.view(-1,self.dim)
        #print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        #print("check model output",x.shape)
        
        return x