import torch
from torch import nn
from model.PositionalEncoder import PositionalEncoder

class ST_CNN(nn.Module):
    def __init__(self, history_len = 35):
        super(ST_CNN,self).__init__()

        self.H_conv1d = nn.Conv1d(
            in_channels=history_len, 
            out_channels=80,
            kernel_size =20, 
            padding= 0,
            stride = 20)

        self.V_conv1d = nn.Conv1d(
            in_channels=5200, 
            out_channels=1000,
            kernel_size =history_len-3, 
            padding= 0,
            stride = 1)      
        self.fc_x2 = nn.Linear(1000,5200)
        self.relu = nn.ReLU(inplace=True)

        self.batchnorm_x1 = nn.BatchNorm1d(80)
        self.batchnorm_x2 = nn.BatchNorm1d(1000)
        self.dropout = nn.Dropout(p=0.2)

        self.fc = nn.Linear(5200, 5200)

        self.positional_encoding_layer = PositionalEncoder(
            dropout=0.01,
            d_model = 5200,
            batch_first = True,
            max_seq_len=100
        )

        self.transEncoder= nn.TransformerEncoderLayer(
            d_model=260, 
            nhead=2, 
            dim_feedforward=2048,
            batch_first=True)

        self.transEncoder_x2= nn.TransformerEncoderLayer(
            d_model=4, 
            nhead=2, 
            dim_feedforward=2048,
            batch_first=True)

    def forward(self, x):
        B,C,L = x.shape

        x = self.positional_encoding_layer(x)
    
        x1 = self.H_conv1d(x)
        x1 = self.transEncoder(x1)
        x1 = self.relu(x1)
        #print(x1.shape)
        x1 = self.batchnorm_x1(x1)
        x1 = self.dropout(x1)
        x1 = x1.reshape(B, 4, 5200)

        x2 = self.V_conv1d(x.permute(0,2,1))
        x2 = self.transEncoder_x2(x2)
        x2 = self.relu(x2)
        x2 = self.batchnorm_x2(x2)
        x2 = self.dropout(x2)
        x2 = x2.permute(0,2,1)

        x2 = self.fc_x2(x2)

        x = x1+x2
        x = self.fc(x)

        return x