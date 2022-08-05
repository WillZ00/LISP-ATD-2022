from random import shuffle
from re import L
import torch
import torch.nn as nn
from torch import MobileOptimizerType
from torch.utils.data import DataLoader
from torch import optim
import os
import time

from data.data_loader import atdDataset
from atd_informer.exp_basic import Exp_Basic
from models.model import Informer

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

class ATD_Transformer():
   # note that exp_basic is passed in the informer model
    def __init__(self,args):
        super(ATD_Transformer, self).__init__(args)

    def _build_model(self):
        print(self.args.dropout)
        print(type(self.args.dropout))


        #mnot
        model = Transformer ( #note that this is not a keyword will probs need to be changed 
            self.args.enc_in,
            self.args.dec_in, 
            self.args.c_out, 
            self.args.seq_len, 
            self.args.label_len,
            self.args.pred_len, 
            self.args.factor,
            self.args.d_model, 
            self.args.n_heads, 
            self.args.e_layers,
            self.args.d_layers, 
            self.args.d_ff,
            self.args.dropout, 
            self.args.attn,
            self.args.embed,
            self.args.freq,
            self.args.activation,
            self.args.output_attention,
            self.args.distil,
            self.args.mix,
            self.device
        ).float()

        return model 

    def _get_data(self, flag):
        args = self.args
        data = atdDataset
        timeenc = 0 if args.embed != "time" else 1

        if flag == "test":
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        
        data_set = Data(
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            #features=args.features,
            #target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )

        print(flag, len(data_set))
        dataset = DataLoader( #this is different from informer bc my dataloadre is called dataset
            data_set,
            batch_size = batch_size,
            shuffle = shuffle_flag
            drop_last = drop_last
        )

        return data_set, dataset 

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr = self.args.learning_rate)
        return model_optim 


    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion