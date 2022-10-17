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

    """
    def fit(self, df:pd.DataFrame, past_covariates=None) -> "InformerForcaster":
        self.df=df
        self.model_list=[]

        for col_index in range(1, len(df.columns), 20):
            self.args.cols=col_index
            informer = atd_informer.ATD_Informer
            for ii in range(self.args.itr):
                # setting record of experiments
                setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(self.args.model, self.args.data, self.args.features, 
                self.args.seq_len, self.args.label_len, self.args.pred_len,
                self.args.d_model, self.args.n_heads, self.args.e_layers, self.args.d_layers, self.args.d_ff, self.args.attn, self.args.factor, self.args.embed, self.args.distil, self.args.mix, self.args.des, ii)

                # set experiments
                exp = informer(self.args, df)
    
                # train
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)
    
                # test
                #print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                #exp.test(setting)

                print(col_index)

                if ii==self.args.itr-1:
                    self.model_list.append(exp)

                
        return self
    
        
        """




    def predict(self, indicies):
        predictions = self.generate_pred(indicies)
        time_idxs=predictions.index
        predictions[predictions<=0]=0
        predictions = predictions.to_numpy()
        predictions = pd.DataFrame(data=predictions, index = time_idxs, columns=self.df.columns)

        return predictions

    """
    def generate_pred(self):
        model_list = self.model_list
        if "timeStamps" in self.df.columns:
            self.df = self.df.drop(["timeStamps"], axis=1)
        #print("cols",self.df.columns)

        name_lst = []
        for i in range(0,self.df.shape[1],20):
            name = self.df.columns[i][0]
            name_lst.append(name)
        current_iter=0
        pred_lst=[]
        #print("name_lst_content", name_lst)
        #print("name_lst_len", len(name_lst))

        #for region_name in name_lst:
            #current_mod = model_list[current_iter]
            #col_lst=[]
            #for i in range(1,21):
            #    col=(region_name, i)
            #    col_lst.append(col)
            #cols=pd.MultiIndex.from_tuples(col_lst)

        #for j in range(4):
            #cur_pred_lst = []
            #for k in range(len(model_list)):
                #current_mod=model_list[k]
                #current_pred = current_mod.predict()
                #current_pred = np.round(current_pred)
                #cur_pred_lst.append(current_pred)
            #current_mod.update_df(np.concatenate(cur_pred_lst))
            
            #print(current_mod.df.tail())
        for k in range(len(model_list)):
            cur_pred_lst = []
            current_mod=model_list[k]
            for j in range(4):
                #current_mod=model_list[k]
                current_pred = current_mod.predict()
                current_pred = np.round(current_pred)
                cur_pred_lst.append(current_pred)
                current_mod.update_df(current_pred)

        
        for i in range(len(model_list)):
            current_mod=model_list[i]
            pred_lst.append(current_mod.df.drop(["timeStamps"], axis=1).tail(4))
            print(pred_lst)
        
        final =pd.concat(pred_lst, axis=1)

        return final
        """
            



    def fit(self, df:pd.DataFrame, past_covariates=None) -> "InformerForcaster":
        self.df=df
        self.model_list=[]

        name_lst=[]
        for i in range(0,self.df.shape[1],20):
            name = self.df.columns[i][0]
            name_lst.append(name)
        col_num=0
        for name in name_lst:
            region_df=self.df[name]
            #print(region_df)
            self.args.cols=col_num
            informer = atd_informer.ATD_Informer
            #print("got here")
            for ii in range(self.args.itr):
                # setting record of experiments
                setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(self.args.model, self.args.data, self.args.features, 
                self.args.seq_len, self.args.label_len, self.args.pred_len,
                self.args.d_model, self.args.n_heads, self.args.e_layers, self.args.d_layers, self.args.d_ff, self.args.attn, self.args.factor, self.args.embed, self.args.distil, self.args.mix, self.args.des, ii)

                # set experiments
                exp = informer(self.args, region_df)
    
                # train
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)
    
                # test
                #print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                #exp.test(setting)

                print(col_num)

                if ii==self.args.itr-1:
                    self.model_list.append(exp)
            col_num+=1
        return self

    def generate_pred(self, indicies):
        forecaster_horizon = len(indicies)
        model_list = self.model_list
        if "timeStamps" in self.df.columns:
            self.df = self.df.drop(["timeStamps"], axis=1)
        #print("cols",self.df.columns)

        name_lst = []
        col_lst=[]
        for i in range(0,self.df.shape[1],20):
            name = self.df.columns[i][0]
            name_lst.append(name)
            
            for j in range(1,21):
                col=(name, j)
                col_lst.append(col)

        current_iter=0
        pred_lst=[]


        for k in range(len(model_list)):
            cur_pred_lst = []
            current_mod=model_list[k]
            for j in range(forecaster_horizon):
                #current_mod=model_list[k]
                current_pred = current_mod.predict()
                current_pred = np.round(current_pred)
                cur_pred_lst.append(current_pred)
                current_mod.update_df(current_pred)

        
        for i in range(len(model_list)):
            current_mod=model_list[i]
            #print("df:",current_mod.df)
            #print("col_idxs", current_mod.df.columns)
            pred_lst.append(current_mod.df.drop(["timeStamps"], axis=1).tail(forecaster_horizon))
            #print(pred_lst)
        
        final =pd.concat(pred_lst, axis=1)

        return final

            