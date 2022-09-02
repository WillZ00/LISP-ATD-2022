import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import gc
from model.SelfAttention import ScaledDotProductAttention


# class CNN_Transformer_Net(nn.Module):
#     def __init__(self):
#         super(CNN_Transformer_Net,self).__init__()

#         self.conv2d = nn.Conv2d(
#             in_channels=1, 
#             out_channels=20,
#             kernel_size =(3, 20), 
#             padding= (1,0),
#             stride = (1,20))
#         self.relu = nn.ReLU(inplace=True)
#         self.batch_norm2d = nn.BatchNorm2d(num_features=20)

#         self.flatten = nn.Flatten()

#         self.conv2d_1 = nn.Conv2d(
#             in_channels=20,
#             out_channels=1,
#             kernel_size=1
#         )

#         self.fc1 = nn.Linear(260, 5200)

#         self.conv2d_2 = nn.Conv2d(
#             in_channels=1,
#             out_channels=1,
#             kernel_size=(4,1),
#             stride=(52,1)
#         )

#         self.gelu = nn.GELU()

#         self.self_att = ScaledDotProductAttention(5200, 100, 100, 10)

#         self.fc2 = nn.Linear(52, 1)

        
#     def forward(self,x):
#         B,C,H,W = x.shape
#         x = self.conv2d(x)
#         x = self.relu(x)
#         x = self.batch_norm2d(x)

#         x = self.conv2d_1(x)
#         x = self.relu(x)
#         x = self.fc1(x)
#         x = x.reshape(B,5200,-1).permute(0,2,1)
#         #print(x.shape)
#         x = self.self_att(x,x,x)
#         x = x.permute(0,2,1)
#         #print(x.shape)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = x.permute(0,2,1)
#         x = x.unsqueeze(dim=1)

#         return x

# class CNN_Transformer_Net(nn.Module):
#     def __init__(self):
#         super(CNN_Transformer_Net,self).__init__()

#         self.conv2d = nn.Conv2d(
#             in_channels=1, 
#             out_channels=20,
#             kernel_size =(3, 20), 
#             padding= (1,0),
#             stride = (1,20))
#         self.relu = nn.ReLU(inplace=True)
#         self.batch_norm2d = nn.BatchNorm2d(num_features=20)

#         self.flatten = nn.Flatten()

#         self.conv2d_1 = nn.Conv2d(
#             in_channels=20,
#             out_channels=1,
#             kernel_size=1
#         )

#         self.fc1 = nn.Linear(260, 5200)

#         self.conv2d_2 = nn.Conv2d(
#             in_channels=1,
#             out_channels=1,
#             kernel_size=(4,1),
#             stride=(52,1)
#         )

#         self.gelu = nn.GELU()

#         self.self_att = ScaledDotProductAttention(5200, 20, 20, 3)

#         self.fc2 = nn.Linear(52, 1)

#         self.transformer = nn.Transformer(
#             d_model=5200, 
#             num_encoder_layers=3, 
#             num_decoder_layers=3,
#             dim_feedforward=256)

        
#     def forward(self,x, y):
#         B,C,H,W = x.shape
#         x = self.conv2d(x)
#         x = self.relu(x)
#         x = self.batch_norm2d(x)
#         #print("0", x.shape)
#         x = self.conv2d_1(x)
#         x = self.relu(x)

#         #print("1", x.shape)
#         x = self.fc1(x)
#         #print("2", x.shape)

#         x = x.squeeze(dim=1)
#         y = y.squeeze(dim=1)

#         x = self.transformer(x, y)

#         x = x.permute(0,2,1)
#         x = self.fc2(x)
#         x = x.permute(0,2,1)
#         return x

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

        self.fc_x1 = nn.Linear(260, 260)
        self.fc_x2 = nn.Linear(20,20)
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
        x1 = x1.transpose(2,3)
        #x2 = x2.reshape(B,260,20, -1)
        x2 = x2.transpose(2,3)

        #print(x1.shape, x2.shape)

        x1 = self.fc1(x1)
        x2 = self.fc1(x2)

        #x1 = x1.squeeze(dim=3)
        #x2 = x2.squeeze(dim=3)

        x1 = x1.transpose(2,3)
        x2 = x2.transpose(2,3)
        #print(x1.shape, x2.shape)

        x1 = x1.transpose(1,2)
        x2 = x2.transpose(1,2)

        x1 = x1.reshape(self.predict_len*B, x1.shape[2], x1.shape[3])
        x2 = x2.reshape(self.predict_len*B, x2.shape[2], x2.shape[3])

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
        x = torch.cat([x1, x2], dim=2)
        
        # x = x1+x2

        # x = self.fc2(x)



        return x