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


# class EMA(nn.Module):
#     def __init__(self, in_size, out_size):
#         super().__init__()
#         self.in_size = in_size
#         self.out_size = out_size
#         self.lambda_ = nn.Parameter(torch.ones(1))
#         self.register_parameter('lambda_', self.lambda_)
#         a = (torch.exp(self.lambda_) - 1)/(torch.exp(self.lambda_) - torch.exp(-self.lambda_*(self.in_size-1)))
#         weights = torch.exp(-torch.arange(0, in_size)*self.lambda_)*a
#         weights = weights.reshape(in_size, 1)
#         self.register_buffer('weights', weights)

        
#     def forward(self, x):
#         for i in range(self.out_size):
#              new_x = torch.matmul(x[:,:,:,-self.in_size:], self.weights)
#              x = torch.concat([x, new_x], dim = -1)
#         return x[:,:,:,-self.out_size:]


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
        
        self.conv2d_1 = nn.Conv2d(
            in_channels=260, 
            out_channels=520,
            kernel_size =(3, 3), 
            padding= 1,
            stride = 1)

        self.conv2d_2 = nn.Conv2d(
            in_channels=520, 
            out_channels=1040,
            kernel_size =(3, 3), 
            padding= 1,
            stride = 1)

        self.conv2d_3 = nn.Conv2d(
            in_channels=1040, 
            out_channels=2080,
            kernel_size =(3, 3), 
            padding= 1,
            stride = 1)

        self.fc1 = nn.Linear(history_len, predict_len)
        self.relu = nn.ReLU(inplace=True)

        self.batch_norm2d_x1 = nn.BatchNorm2d(num_features=20)
        self.batch_norm2d_x2 = nn.BatchNorm2d(num_features=260)

        self.layernorm = nn.LayerNorm((self.history_len,20))
        self.layernorm_x1 = nn.LayerNorm(260)
        self.layernorm_x2 = nn.LayerNorm(20)

        self.transEncoder_x1 = nn.TransformerEncoderLayer(
            d_model=20, 
            nhead=5, 
            dim_feedforward=256,
            batch_first=True)
        self.transEncoder_x2 = nn.TransformerEncoderLayer(
            d_model=260,
            nhead=10, 
            dim_feedforward=2048,
            batch_first=True)

        self.transEncoder = nn.TransformerEncoderLayer(
            d_model=2080,
            nhead=16, 
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
        
        self.fc_x1 = nn.Linear(260, 260)
        self.fc_x2 = nn.Linear(20,20)
        self.rwl = RowWiseLinear(5200,2)
        self.fc2 = nn.Linear(5200,5200)
        self.MLP = MLP(2080, 1040, 520, 260)
        # self.EMA = EMA(self.history_len, self.predict_len)

        self.fc_var = nn.Linear(self.history_len*5200, 5200)
    
    # def forward(self,x1, x2):
    #     B,C,H,W = x1.shape
    #     # x1: B, 1, his_l, 5200(20*260)
    #     x = x1.reshape(B, self.history_len * 5200)
    #     x = self.fc_var(x)
    #     x = x.reshape(B, 1, 5200)

    #     return x


    def forward(self,x1, x2):
        # x1: B, 1, his_l, 5200(20*260)
        # x2: B, 1, his_l, 5200(260*20)
        x = x2
        
        B,C,H,W = x.shape

        x = x.reshape(B, self.history_len, 260, 20).transpose(1,2)
        
        x = self.conv2d_1(x)
        x = self.layernorm(x)
        x = x + self.relu(x)

        x = self.conv2d_2(x)
        x = self.layernorm(x)
        x = x + self.relu(x)

        x = self.conv2d_3(x)
        x = self.layernorm(x)
        x = x + self.relu(x)

        # x: B, 2080, his_l, 20

        x = x.transpose(2,3)

        # x: B, 2080, 20, his_l

        # x = x.permute(0,2,3,1).reshape(B, 20*self.history_len, 2080)
        # x = self.transEncoder(x)
        # x = x.reshape((B, 20, self.history_len, 2080)).permute(0,3,1,2)

        x = self.fc1(x)
        # x = self.EMA(x)

        # # x: B, 2080, 20, pred_l

        x = x.permute(0,3,2,1)

        # x: B, pred_l, 20, 2080

        x = self.MLP(x)

        # x: B, pred_l, 20, 260

        x = x.reshape(B, self.predict_len, 5200)

        return x

    def forward_fmq(self,x1, x2):
        # x1: B, 1, his_l, 5200(20*260)
        # x2: B, 1, his_l, 5200(260*20)
        
        B,C,H,W = x1.shape
        x1 = self.conv2d_x1(x1)
        # x1 = self.batch_norm2d_x1(x1)
        x1 = self.layernorm_x1(x1)
        x1 = self.relu(x1)
        
    
        x2 = self.conv2d_x2(x2)
        # x2 = self.batch_norm2d_x2(x2)
        x2 = self.layernorm_x2(x2)
        x2 = self.relu(x2)

        # x1: B, 20, his_l, 260
        # x2: B, 260, his_l, 20

        x1 = x1.permute(0,2,3,1).reshape(B, self.history_len*260,20)
        x2 = x2.permute(0,2,3,1).reshape(B, self.history_len*20,260)


        # x1 = self.positional_encoding_layer_x1(x1)
        # x2 = self.positional_encoding_layer_x2(x2)

        # x1: B, his_l*260, 20
        # x2: B, his_l*20, 260

        x1 = self.transEncoder_x1(x1)
        x2 = self.transEncoder_x2(x2)

        # x1: B, his_l*260, 20
        # x2: B, his_l*20, 260

        x1 = x1.reshape(B, self.history_len*5200)
        x2 = x2.reshape(B, self.history_len*5200)

        x = torch.concat([x1,x2], dim=-1)
        # x: B, his_l*5200*2

        x = self.MLP(x)

        # x: B, pred_l*5200

        x = x.reshape(B, self.predict_len, 5200)

        # x1 = x1.reshape(B, self.history_len, 260, 20).permute(0,2,3,1)
        # x2 = x2.reshape(B, self.history_len, 20, 260).permute(0,2,3,1)

        # x1 = self.fc1(x1)
        # x2 = self.fc1(x2)

        # # x1: B, 260, 20, pred_l
        # # x2: B, 20, 260, pred_l

        # x1 = x1.permute(0,3,2,1).reshape(B, self.predict_len, 5200)
        # x2 = x2.permute(0,3,1,2).reshape(B, self.predict_len, 5200)

        # x = torch.concat([x1.unsqueeze(-1),x2.unsqueeze(-1)], dim=-1)
        # x = self.rwl(x)

        # # x: B, pred_l, 5200
        # x = self.fc2(x)

        return x