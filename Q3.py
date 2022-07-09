
#importing modules
import pandas as pd
import math
from statsmodels.tsa.ar_model import AutoReg as AR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
#reading a csv file
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
ts = 0.35 # 35% for testing
z = series.values
t_sz = math.ceil(len(z)*ts)
train, test = z[:len(z)-t_sz], z[len(z)-t_sz:]
#for different values of p which indicates the lag
window = [1,5,10,15,25]
rmsevalues = []
mapevalues = []

for i in window:
    model = AR(train, lags=i)
    model_fit = model.fit()  # fit/train the model
    coef = model_fit.params  # Get the coefficients of AR model
    history = train[len(train) - i:]
    history = [history[i] for i in range(len(history))]
    predictions = []  # List to hold the predictions, 1 step at a time
    for t in range(len(test)):
        length = len(history)
        lag = [history[j] for j in range(length - i, length)]
        y = coef[0]  # Initialize to w0
        for d in range(i):
            y += coef[d + 1] * lag[i - d - 1]  # Add other values
        obs = test[t]
        predictions.append(y)  # Append predictions to compute RMSE later
        history.append(obs)  # Append actual test value to history, to be used in next step.
    #finding the root mean square and mape values
    mse = math.sqrt(mean_squared_error(predictions, test))
    mse1 = sum(test) / len(test)
    rmse = ((mse * 100) / (mse1))
    print("rmse values for the lag value", i , "is",rmse)
    ape = mean_absolute_percentage_error(test, predictions)
    mape = ape * 100
    print("mape values for the lag value", i , "is",mape)
    rmsevalues.append(rmse[0])
    mapevalues.append(mape)
#bar graph between the lag values and their rmse values
plt.bar(window,rmsevalues,width = 3)
plt.title("bar plot between rmse and lag values")
plt.xlabel("Lag values")
plt.ylabel("Rmsevalues")
plt.xticks(window)
plt.show()
#bar graph between the lag values and their mape values
plt.bar(window,mapevalues,width = 3)
plt.title("bar plot between mape and lag values")
plt.xlabel("Lag values")
plt.ylabel("Mapevalues")
plt.xticks(window)
plt.show()