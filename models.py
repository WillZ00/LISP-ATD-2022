from dataclasses import dataclass
from typing import Optional
import my_mod as util
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class CnnForecaster:
    training_epochs: int = 500

    def fit(self, data: pd.DataFrame(), past_covariates=None) -> "CnnForecaster":
        # You can ignore `past_covariates`, but include it as an argument.
        x, y_train = util.getMultiDXY(df=data, n_lags=2)
        n_features = 20
        x_train=x.reshape((x.shape[0], x.shape[1], n_features))
        self.model = util.CnnForecastNet().to(device)  # save it for later
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        criterion = nn.MSELoss()
        train = util.myDataset(x_train,y_train)
        train_loader = torch.utils.data.DataLoader(train,batch_size=1,shuffle=False)
        epochs = epochs
        for epoch in range(epochs):
            print('epochs {}/{}'.format(epoch+1,epochs))
            util.Train(self.model, optimizer, train_loader, criterion)
            gc.collect()
        return self


    def predict(self, x: pd.Index) -> pd.DataFrame:
       x_adapted = convert_pdindex_into_whatever_your_model_needs_for_inference(x)
       raw_predictions = self.model.predict(x_adapted)
       predictions = convert_raw_inference_output_into_atd2022_dataframe(raw_predictions)
       return predictions