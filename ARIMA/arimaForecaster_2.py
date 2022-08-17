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
    
    #okay so this is just testing on the test data set, i guess the model requires no training or anything as is stats model 
    for t in range(len(test))
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
       
        yhat = output[0]
        if yhat < 0: 
            yhat = 0
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
