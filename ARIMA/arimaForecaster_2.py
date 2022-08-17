from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error

temp_ls = []

for region in data2:
    temp_region = data2[region]
    #print(temp_region
      
    y = temp_region.values 
    train, test = train_test_split(y, train_size=150)


    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
       
        yhat = output[0]
        xyat = output[1] #note: this is not really x hat just guessing to be next generated predications, I can try and add these prediitions to the model to see how it effects performance
        zhat = output[2]
        khat = output[3]
        if yhat < 0: 
            yhat = 0
        elif xhat < 0: 
            xhat = 0
        elif zhat < 0: 
            zhat = 0
        elif khat < 0: 
            khat = 0
            
            
        #no this is not correct, I want to then 
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
