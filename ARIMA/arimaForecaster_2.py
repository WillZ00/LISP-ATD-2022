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
        
        
def Train(self, data): #not sure about exactly how dataloader would be used here
    #call to data loader clas
    #this is just a subfunc to be called for each sequence of evets for each region
    # i just this this simply would just use model = ARIMA.... and then model fit
    #would have to return a model object
    
def genPred(self, model) #takes as paras model
#likely using trained model and .forecast to extract the next four predicted events
#thats what am trying to figure out what I need to use Dougs classes i dont know if he has that funcxtionality for us

def dataExtrapolate(self, data) # perhaps this would just be the main looper through the data frame to then send to trian and gen pred for each event for each region
