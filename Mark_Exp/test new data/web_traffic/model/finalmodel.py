from operator import concat
import torch
from torch import nn
import numpy as np
from model.models import RowWiseLinear, EMA, MLP, MLP1
import time

class SectionModule(nn.Module):
    def __init__(self, section_len, in_channel, out_channel, history_len, predict_len, groups, attention_first):
        super().__init__()
        self.predict_len=predict_len
        self.history_len = history_len
        self.groups = groups
        self.attention_first = attention_first
        if attention_first:
            self.transEncoder = nn.TransformerEncoderLayer(d_model=history_len, nhead=4)
        self.conv = nn.Conv2d(
            in_channels=in_channel, 
            out_channels=out_channel,
            kernel_size =(3, 3), 
            padding= 1,
            stride = 1,
            groups=groups)        
        self.layernorm = nn.LayerNorm((history_len, section_len))
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(history_len, predict_len)
        self.MLP1 = MLP1(out_channel, in_channel, in_channel)
    
    def forward(self, x):
        # x: B, in_channel, his_l, section_len
        B,C,H,W = x.shape
        if self.attention_first:
            x = x.transpose(2,3).reshape(B*C,W,H)
            x = self.transEncoder(x)
            x = x.transpose(1,2).reshape(B,C,H,W)
        x = self.conv(x)
        x = self.layernorm(x)
        x = x + self.relu(x)
        # x: B out_channel, his_len, section_len
        x = x.transpose(2,3)
        x = self.fc(x)
        x = x.permute(0,3,2,1)
        # x: B, pred_len, section_len, out_channel
        x = self.MLP1(x)
        x = x.permute(0,3,1,2)
        # x: B, in_channel, pred_len, section_len
        return x



class CNN_Transformer_Net(nn.Module):
    def __init__(self, section_structure, data_wedth, history_len=52, predict_len=4):
        super(CNN_Transformer_Net,self).__init__()
        self.predict_len=predict_len
        self.history_len = history_len
        self.section_structure = section_structure
        self.data_wedth = data_wedth
        self.sectional_modules = self._make_sectional_modules(SectionModule, section_structure, history_len, predict_len)
        self.rwl = RowWiseLinear(data_wedth, len(section_structure))


    def _make_sectional_modules(self, layer, section_structure, history_len, predict_len):
        layers = nn.ModuleList([])
        for i in range(len(section_structure)):
            section_catagory_num, in_channel, groups, attention_first = section_structure[i]
            out_channel = 2 * in_channel
            layers.append(layer(section_catagory_num, in_channel, out_channel, history_len, predict_len, groups, attention_first))
        return layers


    def forward(self,xs, valid_idxs):
        y = torch.Tensor([]).to(xs[0].device)
        for i, section_module in enumerate(self.sectional_modules):
            temp = xs[i]
            temp = section_module(temp)
            temp = temp.transpose(1,2).flatten(start_dim=-2)[:,:,valid_idxs[i]]
            y = torch.concat([y, temp.unsqueeze(-1)], dim=-1)
        y = self.rwl(y)
        # y = y.sum(dim=-1)
        return y