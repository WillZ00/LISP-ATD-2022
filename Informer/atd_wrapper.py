import atd2022
import torch
import numpy as np
import pandas as pd
from atd_informer import atd_informer
from utils.tools import dotdict


class InformerForcaster:

    def __init__(self, args):
        self.args=args

    """args = dotdict()
    args.enc_in = 20 # encoder input size
    args.dec_in = 20 # decoder input size
    args.c_out = 20 # output size
    args.factor = 5 # probsparse attn factor
    args.d_model = 512 # dimension of model
    args.n_heads = 8 # num of heads
    args.e_layers = 2 # num of encoder layers
    args.d_layers = 1 # num of decoder layers
    args.d_ff = 2048 # dimension of fcn in model
    args.dropout = 0.05 # dropout
    args.attn = 'prob' # attention used in encoder, options:[prob, full]
    args.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
    args.activation = 'gelu' # activation
    args.distil = True # whether to use distilling in encoder
    args.output_attention = False # whether to output attention in ecoder
    args.mix = True
    args.padding = 0
    args.freq = 'w'
    args.inverse=False
    args.timeenc=0
        #args.cols=1
    args.checkpoints = "/Users/will/Desktop/tmp"


    args.seq_len=5
    args.label_len=3
    args.pred_len=1
    args.batch_size = 1
    args.learning_rate = 0.0001
    args.loss = 'mse'
    args.lradj = 'type1'
    args.use_amp = False

    args.itr=1
    args.train_epochs=1
    args.patience=3"""

    def fit(self, past_covariates=None) -> "InformerForcaster":
        self.model_list=[]

        for col_index in range(1, 5200, 20):
            self.args.cols=col_index
            informer = atd_informer.ATD_Informer
            for ii in range(self.args.itr):
                # setting record of experiments
                setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(self.args.model, self.args.data, self.args.features, 
                self.args.seq_len, self.args.label_len, self.args.pred_len,
                self.args.d_model, self.args.n_heads, self.args.e_layers, self.args.d_layers, self.args.d_ff, self.args.attn, self.args.factor, self.args.embed, self.args.distil, self.args.mix, self.args.des, ii)

                # set experiments
                exp = informer(self.args)
    
                # train
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)
    
                # test
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

                print(col_index)

                if ii==self.args.itr-1:
                    self.model_list.append(exp)

                
        return self

    def predict():

        return pred

    
    def generate_pred(self, full_df:pd.DataFrame):
        model_list = self.model_list

        name_lst = []
        for i in range(0,full_df.shape[1],20):
            name = full_df.columns[i][0]
            name_lst.append(name)
        current_iter=0
        pred_lst=[]
        for region_name in name_lst:
            current_model = model_list[current_iter]
            region_df = full_df[region_name]
            pred = util.pre_trained_region_pred(model=current_model, region_df=region_df, n_lags=2)
            pred = np.round(pred)

            col_lst=[]
            for i in range(1,21):
                col=(region_name, i)
                col_lst.append(col)
        
            cols=pd.MultiIndex.from_tuples(col_lst)
            pred_lst.append(pd.DataFrame(data=pred, columns=cols))
    
        final = pd.concat(pred_lst, axis=1)
        return final
