# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
/*
Program to implement the linear regression using gradient descent.
Developed by: Preethi S

RegisterNumber:  212223230157
*/
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
  X=np.c_[np.ones(len(X1)), X1]

  theta=np.zeros(X.shape[1]).reshape(-1,1)

  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y)
    theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
  return theta

data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
```
![image](https://github.com/user-attachments/assets/1deb79cf-cf39-495d-bbfe-c30fe9fef3cc)

```
X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1: ,-1].values).reshape(-1,1)
print(y)
```

![image](https://github.com/user-attachments/assets/071c12fb-ad2a-43db-8a0f-f29b12f27b2b)

![image](https://github.com/user-attachments/assets/063acc9f-1f21-4d84-b163-9d93878f7838)
```
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
```
![image](https://github.com/user-attachments/assets/147157a0-fcf8-4a91-8056-f68f16139a2f)

![image](https://github.com/user-attachments/assets/81b1e159-2b4a-4b1e-a3ca-f9a832727862)

```
theta=linear_regression(X1_Scaled, Y1_Scaled)

new_data=np.array([165349.2,136897,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```
![image](https://github.com/user-attachments/assets/6213b972-a570-4313-87fa-499a6414eb21)





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
