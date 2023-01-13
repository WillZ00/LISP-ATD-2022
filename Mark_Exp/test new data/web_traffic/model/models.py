import torch
from torch import nn


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