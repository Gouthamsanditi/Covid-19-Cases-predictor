
#importing modules
import pandas as pd
import math
from statsmodels.tsa.ar_model import AutoReg as AR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
#reading a csv file
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
#splitting of the data into  train and test
ts = 0.35 # 35% for testing
z = series.values
t_sz = math.ceil(len(z)*ts)
train, test = z[:len(z)-t_sz], z[len(z)-t_sz:]
#taking a lag value of 5
window = 5
model = AR(train, lags= window)
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model
print(coef)
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
coefficients = np.round(coef, decimals = 3)
print("coefficients for the AR model are", coefficients)
#scatter plot between the test data and the predicted test data
plt.scatter(test,predictions)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Scatter plot between the actual and predicted values")
plt.show()
#line plot of original test and the predicted test data
plt.plot(test)
plt.plot(predictions)
plt.legend(["test","predictions"])
plt.ylabel("Actual values and predicted values")
plt.xlabel("Index")
plt.title("line plot of  actual and predicted values")
plt.show()
#finding of the rmse and mape values for predicted data and the original data
mse = math.sqrt(mean_squared_error(predictions,test))
mse1 = sum(test)/len(test)
rmse = ((mse*100)/(mse1))
print(rmse)
ape = mean_absolute_percentage_error(test,predictions)
mape = ape*100
print(mape)