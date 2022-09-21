from torch import nn, sqrt
import torch
import sys
from math import sqrt
sys.path.append('.')
from model.conv.MBConv import MBConvBlock
from model.attention.SelfAttention import ScaledDotProductAttention


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


class CoAtNet(nn.Module):
    def __init__(self,in_ch,image_size,out_chs=[64,96,192,384,768], history_len = 20):
        super().__init__()
        self.out_chs=out_chs
        self.maxpool2d=nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool1d = nn.MaxPool1d(kernel_size=1, stride=2)
        self.history_len = history_len

        self.s0=nn.Sequential(
            nn.Conv2d(in_ch,in_ch,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=260),
            nn.Conv2d(in_ch,in_ch,kernel_size=3,padding=1)
        )
        self.mlp0=nn.Sequential(
            nn.Conv2d(in_ch,out_chs[0],kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(out_chs[0],out_chs[0],kernel_size=1)
        )
        
        self.s1=MBConvBlock(ksize=3,input_filters=out_chs[0],output_filters=out_chs[0],image_size=image_size//2)
        self.mlp1=nn.Sequential(
            nn.Conv2d(out_chs[0],out_chs[1],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_chs[1],out_chs[1],kernel_size=1)
        )

        self.s2=MBConvBlock(ksize=3,input_filters=out_chs[1],output_filters=out_chs[1],image_size=image_size//4)
        self.mlp2=nn.Sequential(
            nn.Conv2d(out_chs[1],out_chs[2],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_chs[2],out_chs[2],kernel_size=1)
        )

        self.s3=ScaledDotProductAttention(out_chs[2],out_chs[2]//6,out_chs[2]//6,6)
        self.mlp3=nn.Sequential(
            nn.Linear(out_chs[2],out_chs[3]),
            nn.ReLU(),
            nn.Linear(out_chs[3],out_chs[4])
        )

        self.s4=ScaledDotProductAttention(out_chs[3],out_chs[3]//6,out_chs[3]//6,6)
        self.mlp4=nn.Sequential(
            nn.Linear(out_chs[3],out_chs[4]),
            nn.ReLU(),
            nn.Linear(out_chs[4],out_chs[4])
        )
        self.fc1 = nn.Linear(out_chs[4], 20)
        self.fc = nn.Linear(5200,5200)
        self.relu = nn.ReLU()

        self.EMA = EMA(self.history_len, 20)
        self.fc_rd = nn.Linear(self.history_len, 20)

    def forward(self, y, y1) :
        B,C,H,W=y.shape
        
        y = y.reshape(B, self.history_len, 260, 20).transpose(1,2)
        y = y.transpose(2,3)
        y = self.EMA(y)
        y = y.transpose(2,3)

        #stage0
        y=self.mlp0(self.s0(y))
        y=self.maxpool2d(y)
        #stage1
        y=self.mlp1(self.s1(y))
        y=self.maxpool2d(y)
        #stage2
        #y=self.mlp2(self.s2(y))
        y = self.mlp2(y)
        y=self.maxpool2d(y)
        #stage3
        y=y.reshape(B,self.out_chs[2],-1).permute(0,2,1) #B,N,C
        y=self.mlp3(self.s3(y,y,y))
        y=self.maxpool1d(y.permute(0,2,1)).permute(0,2,1)
        #print(y.shape)
        #stage4
        #y=self.mlp4(self.s4(y,y,y))
        #y = self.mlp4(y)
        y=self.maxpool1d(y.permute(0,2,1))
        N=y.shape[-1]
        y=y.reshape(B,self.out_chs[4],int(sqrt(N)),int(sqrt(N)))
        #print(y.shape)
        #y = y.squeeze()
        #y = self.fc1(y)
        y = y.reshape(B, 1, 5200)

        #y = self.relu(y)

        y = self.fc(y)

        return y

if __name__ == '__main__':
    x=torch.randn(1,3,224,224)
    coatnet=CoAtNet(3,224)
    y=coatnet(x)
    print(y.shape)