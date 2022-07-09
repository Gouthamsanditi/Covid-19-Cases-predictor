
#importing modules
import pandas as pd
import math
import numpy as  np
from statsmodels.tsa.ar_model import AutoReg as AR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
data  =  pd.read_csv('daily_covid_cases.csv')
traindata, testdata = train_test_split(data,test_size = 0.35, shuffle = False)
new_cases = traindata["new_cases"]
X = 2/(math.sqrt(len(traindata)))
print(X)
corre = []
p = []
for i in range(1,len(traindata)):
    x = new_cases.shift(i)
    corr = new_cases.corr(x)
    if corr > X:
        p.append([i])
        corre.append(corr)
plt.plot(p,corre)
plt.xlabel("lag values")
plt.ylabel("correlation values")
plt.title("line plot between the lag and correlation values")
plt.show()
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
ts = 0.35 # 35% for testing
z = series.values
t_sz = math.ceil(len(z)*ts)
train, test = z[:len(z)-t_sz], z[len(z)-t_sz:]

#by elbow rule heurstic value is 77
window = 77
model = AR(train, lags= window)
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = [] # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    y = coef[0] # Initialize to w0
    for d in range(window):
        y += coef[d+1] * lag[window-d-1] # Add other values
    obs = test[t]
    predictions.append(y) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.
    #finding rmse and mape values
mse = math.sqrt(mean_squared_error(predictions,test))
mse1 = sum(test)/len(test)
rmse = ((mse*100)/(mse1))
print(rmse.round(3))
ape = mean_absolute_percentage_error(test,predictions)
mape = ape*100
print(mape.round(3))

