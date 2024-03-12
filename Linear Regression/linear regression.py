import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

# reading the csv in data
data  = pd.read_csv("Advertisement.csv")

# dropping the unnamed variable defined by system against the blank value to clean the data
data.drop(['Unnamed: 0'],axis=1)

#setting the size of graph
plt.figure(figsize=(16,8))

#setting the axises and colour of scatter
plt.scatter(data['tv'], data['sales'], c='black')

X = data['tv'].values.reshape(-1,1)

Y = data['sales'].values.reshape(-1,1)

reg = LinearRegression()

reg.fit(X,Y)

predictions = reg.predict(X)

plt.figure(figsize=(16, 8))

plt.scatter(
    data['tv'],
    data['sales'],
    c='black'
)

plt.plot(
    data['tv'],
    predictions,
    c='blue',
    linewidth=2
)

plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
plt.show()