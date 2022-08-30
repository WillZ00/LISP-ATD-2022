import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import gc

# class CNN_ForecastNet(nn.Module):
#     def __init__(self):
#         super(CNN_ForecastNet,self).__init__()

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

    
        
#     def forward(self,x):
#         x = self.conv2d(x)
#         x = self.relu(x)
#         x = self.batch_norm2d(x)

#         x = self.conv2d_1(x)
#         x = self.relu(x)
#         x = self.fc1(x)
        
#         return x


class CNN_ForecastNet(nn.Module):
    def __init__(self):
        super(CNN_ForecastNet,self).__init__()

        self.conv2d = nn.Conv2d(
            in_channels=1, 
            out_channels=20,
            kernel_size =(3, 20), 
            padding= (1,0),
            stride = (1,20))
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm2d = nn.BatchNorm2d(num_features=20)

        self.flatten = nn.Flatten()

        self.conv2d_1 = nn.Conv2d(
            in_channels=20,
            out_channels=1,
            kernel_size=1
        )

        self.fc1 = nn.Linear(260, 5200)

        self.conv2d_2 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(4,1),
            stride=(52,1)
        )

        self.gelu = nn.GELU()

        

    
        
    def forward(self,x):
        x = self.conv2d(x)
        x = self.gelu(x)
        x = self.batch_norm2d(x)

        x = self.conv2d_1(x)
        x = self.gelu(x)
        x = self.fc1(x)
        x = self.conv2d_2(x)
        x = self.gelu(x)
        
        return x