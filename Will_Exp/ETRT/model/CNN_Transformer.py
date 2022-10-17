from operator import concat
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from statsmodels.tsa.api import VAR
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


class EMA(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.diff_size = in_size - out_size + 1
        self.lambda_ = nn.Parameter(torch.ones(1))
        self.register_parameter('lambda_', self.lambda_)

        
    def forward(self, x):
        weights = (torch.exp(-(torch.arange(0, self.diff_size)).to('cuda')*self.lambda_)*(torch.exp(self.lambda_) - 1)/(torch.exp(self.lambda_) - torch.exp(-self.lambda_*(self.diff_size-1))))
        weights = torch.concat([weights, torch.zeros(self.out_size-1).to('cuda')])
        weights = torch.concat([weights.roll(shifts=i).unsqueeze(dim=1) for i in range(self.out_size)], dim=1).to('cuda')
        x = torch.matmul(x, weights)
        return x


class MLP(torch.nn.Module): 
    def __init__(self, in_channel, hidden_channel1, hidden_channel2, out_channel):
        super(MLP,self).__init__() 
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = torch.nn.Linear(in_channel, hidden_channel1)  # 第一个隐含层  
        self.fc2 = torch.nn.Linear(hidden_channel1, hidden_channel2)  
        self.fc3 = torch.nn.Linear(hidden_channel2, out_channel)   # 输出层
        
    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP1(torch.nn.Module): 
    def __init__(self, in_channel, hidden_channel, out_channel):
        super(MLP1,self).__init__() 
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = torch.nn.Linear(in_channel, hidden_channel)  # 第一个隐含层  
        self.fc2 = torch.nn.Linear(hidden_channel, out_channel)   # 输出层
        
    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_Transformer_Net(nn.Module):
    def __init__(self, history_len=52, predict_len=4):
        super(CNN_Transformer_Net,self).__init__()
        self.predict_len=predict_len
        self.history_len = history_len
        
        self.conv2d_1 = nn.Conv2d(
            in_channels=260, 
            out_channels=520,
            kernel_size =(3, 3), 
            padding= 1,
            stride = 1)

        self.conv2d_2 = nn.Conv2d(
            in_channels=20, 
            out_channels=260,
            kernel_size =(3, 13), 
            padding= (1,0),
            stride = (1,13))
        
        self.conv2d_3 = nn.Conv2d(
            in_channels=1, 
            out_channels=520,
            kernel_size =(3, 260), 
            padding= (1,0),
            stride = (1,260))

        

        self.fc1 = nn.Linear(history_len, predict_len)
        self.relu = nn.ReLU(inplace=True)

        self.cnn_mlp=nn.Sequential(
            nn.Conv2d(self.history_len,self.predict_len,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(self.predict_len,self.predict_len,kernel_size=3,padding=1)
        )

        self.batch_norm2d = nn.BatchNorm2d(num_features=520)

        self.layernorm = nn.LayerNorm((self.history_len,20))
        self.layernorm2 = nn.LayerNorm((self.history_len,20))
        self.layernorm_1 = nn.LayerNorm((self.history_len,20))

        self.transEncoder = nn.TransformerEncoderLayer(
            d_model=260,
            nhead=20, 
            dim_feedforward=2048,
            batch_first=True)

        self.positional_encoding_layer_x1 = PositionalEncoder(
            dropout=0.01,
            d_model = 20,
            batch_first = True,
            max_seq_len=10000
        )
        self.positional_encoding_layer_x2 = PositionalEncoder(
            dropout=0.01,
            d_model = 260,
            batch_first = True,
            max_seq_len=10000
        )
        
        # self.rwl = RowWiseLinear(5200,2)
        self.MLP = MLP(2080, 1040, 520, 260)
        self.MLP1 = MLP1(1040, 520, 260)
        self.EMA = EMA(self.history_len, self.predict_len)
        # self.fc_var = nn.Linear(self.history_len*5200, 5200)

        self.attention=nn.MultiheadAttention(
            embed_dim=4080,
            num_heads=2,
            batch_first=True,
        )
        self.conv2d_4 = nn.Conv2d(
            in_channels=1040, 
            out_channels=2040,
            kernel_size =(3, 3), 
            padding= (1,1),
            stride = (1))

        self.conv2d_5 = nn.Conv2d(
            in_channels=2040, 
            out_channels=5200,
            kernel_size =(1, self.history_len), 
            padding= (0),
            stride = (1))

        self.maxpool2d=nn.MaxPool2d(kernel_size=2,stride=2)
        # self.encoderLayer = nn.TransformerEncoderLayer(
        #     d_model=20,
        #     nhead=2, 
        #     dim_feedforward=512,
        #     batch_first=True)


        self.conv1d_6 = nn.Conv1d(
            in_channels=4080,
            out_channels=5200,
            kernel_size = 3,
            padding=1,
            stride = 1
        )
        self.conv1d_7 = nn.Conv1d(
            in_channels=5200,
            out_channels=5200,
            kernel_size = 3,
            padding=0,
            stride =1
        )
        self.conv1d_8 = nn.Conv1d(
            in_channels=5200,
            out_channels=5200,
            kernel_size = 2,
            padding=0
        )
        self.reduc_fc = nn.Linear(20,1)
        self.final_fc = nn.Linear(5200,5200)

    def forward(self,x1, x2):
        # x1: B, 1, his_l, 5200(260*20)
        # x2: B, 1, his_l, 5200(20*260)
        B,C,H,W = x1.shape

        x1 = x1.reshape(B, self.history_len, 260, 20).transpose(1,2)

        x = self.conv2d_1(x1)
        x = self.layernorm(x)
        x = x+ self.relu(x)

        # x: B, 520, his_l, 20

        x_2 = self.conv2d_3(x2)
        x_2 = self.layernorm2(x_2)
        x_2 = x_2 + self.relu(x_2)
        # print("x1:", x1.shape)
        # print("x2:", x2.shape)

        # # x2:B, 260, his_l, 20
        x = torch.concat([x, x_2], dim=1)

        x = x.transpose(2,3)

        # x: B, 2080, 20, his_l


        # x = x.permute(0,2,3,1).reshape(B, 20*self.history_len, -1)
        # x = self.transEncoder(x)
        # x = x.reshape((B, 20, self.history_len, -1)).permute(0,3,1,2)

        x = self.conv2d_4(x)

        # print("post_conv", x.shape)
        x = self.relu(x)


        x = self.conv2d_5(x)

        x = self.relu(x)
        
        # print("post_conv5", x.shape)

        x = x.reshape(B,5200,20)

        x = self.reduc_fc(x)

        # print("post_sqz", x.shape)

        # x = x.permute(0,2,1)

        # x,w = self.attention(x,x,x)

        # x = x.permute(0,2,1)

        # x = self.encoderLayer(x)

        # print("post_enc", x.shape)


        # x = self.conv1d_6(x)

        # # print("post_conv6", x.shape)

        # for i in range(9):
        #     x = self.conv1d_7(x)
        #     # x = x.relu()

        # # print("post_conv7", x.shape)

        # x = self.conv1d_8(x)
        # x = x.relu()
        # # print("post_conv8", x.shape)
        
        x = x.reshape(B,1,5200)

        x = self.final_fc(x)

        return x