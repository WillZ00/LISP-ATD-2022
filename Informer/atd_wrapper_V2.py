import atd2022
import torch
import numpy as np
import pandas as pd
from atd_informer.atd_informer_V2 import ATD_Informer_V2
from utils.tools import dotdict


class InformerForcaster_V2:

    def __init__(self, args):
        self.args=args


    def fit(self, df:pd.DataFrame, past_covariates=None) -> "InformerForcaster":
        self.df=df
        #self.model = ATD_Informer_V2(self.args, self.df)

        name_lst=[]
        for i in range(0,self.df.shape[1],20):
            name = self.df.columns[i][0]
            name_lst.append(name)
        col_num=0

        for ii in range(self.args.itr):
            # setting record of experiments
            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(self.args.model, self.args.data, self.args.features, 
            self.args.seq_len, self.args.label_len, self.args.pred_len,
            self.args.d_model, self.args.n_heads, self.args.e_layers, self.args.d_layers, self.args.d_ff, self.args.attn, self.args.factor, self.args.embed, self.args.distil, self.args.mix, self.args.des, ii)

            # set experiments
            exp = ATD_Informer_V2(self.args, self.df)

            # train
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            # test
            #print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            #exp.test(setting)

            #print(col_num)

            self.model = exp
        return self

    def generate_pred(self, indicies):
        forecaster_horizon = len(indicies)

        model = self.model
        if "timeStamps" in self.df.columns:
            self.df = self.df.drop(["timeStamps"], axis=1)


        # for j in range(forecaster_horizon):
        #     pred = model.predict()
        #     pred = np.round(pred)
        #     model.update_df(pred)
        pred = model.predict()
        print(pred)
        print(type(pred))
        print(pred.shape)
        

        final = model.df.drop(["timeStamps"], axis=1).tail(forecaster_horizon)
        return final

    def predict(self, indicies):
        predictions = self.generate_pred(indicies)
        time_idxs=predictions.index
        predictions[predictions<=0]=0
        predictions = predictions.to_numpy()
        predictions = pd.DataFrame(data=predictions, index = time_idxs, columns=self.df.columns)

        return predictions