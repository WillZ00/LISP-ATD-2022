import atd2022
import torch
import numpy as np
import pandas as pd
from atd_CoAtNet.atd_CoAtNet import ATD_CoAtNet
from utils.tools import dotdict


class CoAtNetForecaster():

    def __init__(self, args):
        self.args = args
    
    def fit(self, df:pd.DataFrame, past_covariates=None) -> "CoAtNetForecaster":
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
            CoAtNet = ATD_CoAtNet
            #print("got here")
            for ii in range(self.args.itr):
                # setting record of experiments
                setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(self.args.model, self.args.data, self.args.features, 
                self.args.seq_len, self.args.label_len, self.args.pred_len,
                self.args.d_model, self.args.n_heads, self.args.e_layers, self.args.d_layers, self.args.d_ff, self.args.attn, self.args.factor, self.args.embed, self.args.distil, self.args.mix, self.args.des, ii)

                # set experiments
                exp = CoAtNet(self.args, region_df)

                # train
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print(col_num)

                if ii==self.args.itr-1:
                    self.model_list.append(exp)
            col_num+=1
        return self


    def predict(self, indicies):
        predictions = self.generate_pred(indicies)
        time_idxs=predictions.index
        predictions[predictions<=0]=0
        predictions = predictions.to_numpy()
        predictions = pd.DataFrame(data=predictions, index = time_idxs, columns=self.df.columns)

        return predictions



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
