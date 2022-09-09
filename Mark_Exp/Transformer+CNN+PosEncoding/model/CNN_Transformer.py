import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import gc
from model.SelfAttention import ScaledDotProductAttention
from model.PositionalEncoder import PositionalEncoder

class RowWiseLinear(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.weights = nn.Parameter(torch.ones(height, 1, width))
        self.register_parameter('weights', self.weights)
        # self.weights = nn.Parameter(weights)
        # self.weights = torch.ones(height, 1, width).to('cuda')
        # self.register_buffer('mybuffer', self.weights)

        
    def forward(self, x):
        x_unsqueezed = x.unsqueeze(-1)
        w_times_x = torch.matmul(self.weights, x_unsqueezed)
        return w_times_x.squeeze()

class CNN_Transformer_Net(nn.Module):
    def __init__(self, history_len=52, predict_len=4):
        super(CNN_Transformer_Net,self).__init__()
        self.predict_len=predict_len
        self.history_len = history_len

        self.conv2d_x1 = nn.Conv2d(
            in_channels=1, 
            out_channels=20,
            kernel_size =(3, 20), 
            padding= (1,0),
            stride = (1,20))

        self.conv2d_x2 = nn.Conv2d(
            in_channels=1, 
            out_channels=260,
            kernel_size =(3, 260), 
            padding= (1,0),
            stride = (1,260))

        self.fc1 = nn.Linear(history_len, predict_len)
        self.relu = nn.ReLU(inplace=True)

        self.batch_norm2d_x1 = nn.BatchNorm2d(num_features=20)
        self.batch_norm2d_x2 = nn.BatchNorm2d(num_features=260)

        self.transEncoder_x1 = nn.TransformerEncoderLayer(
            d_model=260, 
            nhead=10, 
            dim_feedforward=2048,
            batch_first=True)
        self.transEncoder_x2 = nn.TransformerEncoderLayer(
            d_model=20,
            nhead=5, 
            dim_feedforward=256,
            batch_first=True)

        self.positional_encoding_layer_x1 = PositionalEncoder(
            dropout=0.01,
            d_model = 260,
            batch_first = True,
            max_seq_len=100
        )
        self.positional_encoding_layer_x2 = PositionalEncoder(
            dropout=0.01,
            d_model = 20,
            batch_first = True,
            max_seq_len=100
        )

        self.fc_x1 = nn.Linear(260, 260)
        self.fc_x2 = nn.Linear(20,20)
        self.rwl = RowWiseLinear(5200,2)
        self.fc2 = nn.Linear(5200,5200)

    def forward(self,x1, x2):
        
        B,C,H,W = x1.shape
        x1 = self.conv2d_x1(x1)
        x1 = self.relu(x1)
        x1 = self.batch_norm2d_x1(x1)
    
        x2 = self.conv2d_x2(x2)
        x2 = self.relu(x2)
        x2 = self.batch_norm2d_x2(x2)

        #x1 = x1.reshape(B,20,260, -1)
        # x1 = x1.transpose(2,3)
        #x2 = x2.reshape(B,260,20, -1)
        # x2 = x2.transpose(2,3)

        x1 = self.positional_encoding_layer_x1(x1)
        x2 = self.positional_encoding_layer_x2(x2)

        #x1 = self.fc1(x1)
        #x2 = self.fc1(x2)

        #x1 = x1.squeeze(dim=3)
        #x2 = x2.squeeze(dim=3)

        # x1 = x1.permute(0,3,1,2)
        # x2 = x2.permute(0,3,1,2)

        x1 = x1.reshape(20*B, self.history_len, 260)
        x2 = x2.reshape(260*B, self.history_len, 20)
        # print(x1.shape, x2.shape)
        x1 = self.transEncoder_x1(x1)
        #x1 = self.transEncoder_x1(x1)
        #x1 = self.transEncoder_x1(x1)
        x2 = self.transEncoder_x2(x2)
        #x2 = self.transEncoder_x2(x2)
        #x2 = self.transEncoder_x2(x2)

        x1 = x1.transpose(1,2)
        x2 = x2.transpose(1,2)

        x1 = self.fc1(x1)
        x2 = self.fc1(x2)

        x1 = x1.transpose(1,2)
        x2 = x2.transpose(1,2)

        x1 = self.relu(x1)
        x2 = self.relu(x2)

        # x1 = self.fc_x1(x1)
        # x2 = self.fc_x2(x2)

        x1 = x1.reshape(B, 20, self.predict_len, 260)
        x2 = x2.reshape(B, 260, self.predict_len, 20)
        
        x1 = x1.transpose(1,2)
        x2 = x2.transpose(1,2)

        x2 = x2.transpose(2,3)

        x1 = x1.reshape(B, self.predict_len, 5200)
        x2 = x2.reshape(B, self.predict_len, 5200)
        #print(x1.shape, x2.shape)

        x = torch.concat([x1.unsqueeze(-1),x2.unsqueeze(-1)], dim=-1)
        x = self.rwl(x)

        # x = x1+x2
        # x = self.fc2(x)
        

        return x