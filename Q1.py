
#importing modules
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sm
#reading a csv file
data = pd.read_csv("daily_covid_cases.csv")
d1 = data["Date"]
d2 = data["new_cases"]
correlation_values = []
#creating a line plot of new cases with respect to the index of the date
plt.plot(d1,d2,color = "red")
plt.title("Number of covid cases with at a particular day")
plt.xlabel("Index of the day")
plt.ylabel("Number of covid cases")
plt.show()
#shifting of the data by one day lag
lag_cases = d2.shift(1)
#correlation between the original data and the one day time lag data
corr = d2.corr(lag_cases)
print(corr)
#scatter plot between the lag cases and the original data new_cases
plt.scatter(lag_cases,d2)
plt.title("Scatter plot between the original time sequence and lagged time sequence by 1")
plt.xlabel("the original time sequence")
plt.ylabel("lagged time sequence by 1")
plt.show()
#Generating the multiple lag series with different values of p and finding correlation between the original data and lagged series
p = [1,2,3,4,5,6]
for i in p:
    lagcases = d2.shift(i)
    corr1 = d2.corr(lagcases)
    print("the autocorrelation between the generated", i,  "day lag time sequence and the given time sequence is" ,corr1)
    correlation_values.append(corr1)
#line plot of correlation values with respect to their lag values
plt.plot(p,correlation_values,color = 'black')
plt.title("a line plot between obtained correlation coefficients  and lagged values")
plt.xlabel("lagged values")
plt.ylabel("correlation coefficients")
plt.show()
#line plot using the autocorrelation function with different lag values
sm.plot_acf(d2, lags=6)
plt.show()