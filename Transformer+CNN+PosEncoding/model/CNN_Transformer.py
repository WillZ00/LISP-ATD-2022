import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import gc
from model.SelfAttention import ScaledDotProductAttention
from model.PositionalEncoder import PositionalEncoder

class CNN_Transformer_Net(nn.Module):
    def __init__(self, history_len=52, predict_len=4):
        super(CNN_Transformer_Net,self).__init__()
        self.predict_len=predict_len

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
            nhead=10, 
            dim_feedforward=1024,
            batch_first=True)

        self.positional_encoding_layer_x1 = PositionalEncoder(
            dropout=0.01,
            d_model = 260,
            batch_first = True
        )
        self.positional_encoding_layer_x2 = PositionalEncoder(
            dropout=0.01,
            d_model = 20,
            batch_first = True
        )

        self.fc_x1 = nn.Linear(260, 260)
        self.fc_x2 = nn.Linear(20,20)
        self.fc2 = nn.Linear(2,5200)

    def forward(self,x1, x2):
        
        B,C,H,W = x1.shape
        x1 = self.conv2d_x1(x1)
        x1 = self.relu(x1)
        x1 = self.batch_norm2d_x1(x1)
    
        x2 = self.conv2d_x2(x2)
        x2 = self.relu(x2)
        x2 = self.batch_norm2d_x2(x2)

        #x1 = x1.reshape(B,20,260, -1)
        x1 = x1.transpose(2,3)
        #x2 = x2.reshape(B,260,20, -1)
        x2 = x2.transpose(2,3)

        print(x1.shape, x2.shape)

        x1 = self.positional_encoding_layer_x1(x1)
        x2 = self.positional_encoding_layer_x2(x2)

        print(x1.shape, x2.shape)

        #x1 = self.fc1(x1)
        #x2 = self.fc1(x2)

        #x1 = x1.squeeze(dim=3)
        #x2 = x2.squeeze(dim=3)

        x1 = x1.transpose(2,3)
        x2 = x2.transpose(2,3)
        #print(x1.shape, x2.shape)

        x1 = x1.transpose(1,2)
        x2 = x2.transpose(1,2)

        x1 = x1.reshape(20*B, x1.shape[2], x1.shape[3])
        x2 = x2.reshape(260*B, x2.shape[2], x2.shape[3])

        x1 = self.transEncoder_x1(x1)
        #x1 = self.transEncoder_x1(x1)
        #x1 = self.transEncoder_x1(x1)
        x2 = self.transEncoder_x2(x2)
        #x2 = self.transEncoder_x2(x2)
        #x2 = self.transEncoder_x2(x2)

        x1 = self.fc_x1(x1)
        x2 = self.fc_x2(x2)
        x1 = x1.transpose(1,2)
        x1 = x1.reshape(B, self.predict_len, 5200)
        x2 = x2.reshape(B, self.predict_len, 5200)

        x1 = self.relu(x1)
        x2 = self.relu(x2)
        #print(x1.shape, x2.shape)

        # x = torch.cat([x1, x2], dim=2)
        #35,1,4,10400
        # x = x.reshape(B, x.shape[1], 5200, 2)
        #print(x.shape)
        # x = x1+x2
        # x = self.fc2(x)
        # x = x.diagonal(dim1=2,dim2=3)
        x = x1*x2

        return x