import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import gc
from model.SelfAttention import ScaledDotProductAttention

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

# Working 2D-CNN 1
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

#         self.conv2d_2 = nn.Conv2d(
#             in_channels=1,
#             out_channels=1,
#             kernel_size=(4,1),
#             stride=(52,1)
#         )

#         self.gelu = nn.GELU()

        
#     def forward(self,x):
#         x = self.conv2d(x)
#         x = self.relu(x)
#         x = self.batch_norm2d(x)

#         x = self.conv2d_1(x)
#         x = self.relu(x)
#         x = self.fc1(x)
#         x = self.conv2d_2(x)
#         x = self.relu(x)
        
#         return x


# Working 2d cnn with self attention
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

#         print("0", x.shape)

#         x = self.conv2d_1(x)
#         x = self.relu(x)
#         print("1", x.shape)
#         x = self.fc1(x)
#         print("2", x.shape)



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
        self.batch_norm2d_1 = nn.BatchNorm2d(num_features=1)

        self.flatten = nn.Flatten()

        self.conv2d_1 = nn.Conv2d(
            in_channels=20,
            out_channels=1,
            kernel_size=1
        )

        self.fc1 = nn.Linear(52, 1)
        self.fc2 = nn.Linear(5200,5200)

        self.conv2d_2 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(4,1),
            stride=(52,1)
        )

    
        
    def forward(self,x):
        B,C,H,W = x.shape
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.batch_norm2d(x)
        #print(x.shape)

        x = x.reshape(B, 20, -1, H)
        #print(x.shape)
        x = self.fc1(x)

        x = self.relu(x)


        x = x.reshape(B,1,20, 260)
        x = x.reshape(B, 1, 5200)

        x = self.fc2(x)

        
        return x







# class CNN_ForecastNet(nn.Module):
#     def __init__(self):
#         super(CNN_ForecastNet,self).__init__()

#         self.conv2d = nn.Conv2d(
#             in_channels=1, 
#             out_channels=1,
#             kernel_size =(3, 3), 
#             padding= (1,1),
#             stride = 1)
#         self.relu = nn.ReLU(inplace=True)
#         self.batch_norm2d = nn.BatchNorm2d(num_features=20)
#         self.batch_norm2d_1 = nn.BatchNorm2d(num_features=1)

#         self.flatten = nn.Flatten()

#         self.conv2d_1 = nn.Conv2d(
#             in_channels=20,
#             out_channels=1,
#             kernel_size=1
#         )

#         self.fc1 = nn.Linear(5200, 5200)

#         self.conv2d_2 = nn.Conv2d(
#             in_channels=1,
#             out_channels=1,
#             kernel_size=(4,1),
#             stride=(52,1)
#         )

        
#     def forward(self,x):
#         x = self.conv2d(x)
#         x = self.batch_norm2d_1(x)
#         x = self.relu(x)

#         x = self.conv2d_2(x)
#         x = self.batch_norm2d_1(x)
#         x = self.relu(x)

#         x = self.fc1(x)
#         x = self.relu(x)
#         return x