## Implementation-of-Linear-Regression-Using-Gradient-Descent
# AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

# Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
Algorithm
Import numpy, pandas, and StandardScaler from sklearn.preprocessing.
Read '50_Startups.csv' into a DataFrame (data) using pd.read_csv().
Extract features (X) and target variable (y) from the DataFrame. Convert features to a numpy array (x1) and target variable to a numpy array (y). Scale the features using StandardScaler(). Linear Regression Function:
Define linear_regression(X1, y) function for linear regression. Add a column of ones to features for the intercept term. Initialize theta as a zero vector. Implement gradient descent to update theta. Model Training and Prediction:
Call linear_regression function with scaled features (x1_scaled) and target variable (y). Prepare new data for prediction by scaling and reshaping. Use the optimized theta to predict the output for new data. Print Prediction:
Inverse transform the scaled prediction to get the actual predicted value. Print the predicted value
# Program:
Program to implement the linear regression using gradient descent.
Developed by:Preethi S
RegisterNumber:  212223230157
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


X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1: ,-1].values).reshape(-1,1)
print(y)


X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)


theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data=np.array([165349.2,136897,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```
# Output:
![363420043-ad48d1d5-a1a8-41a5-b4f3-c6613ce088e2](https://github.com/user-attachments/assets/338c6a61-898d-4fd6-812c-d44d187b7f84)

X
![382651061-7650568c-1516-4917-a9b9-f5d93635b967](https://github.com/user-attachments/assets/0a6f07c4-66b0-466b-91f7-f638ef23e03b)

Y
![382651132-615fe446-ddef-4f0c-92b0-884ebe1e7e80](https://github.com/user-attachments/assets/62c95620-9deb-43ad-bc88-800d8a59a111)

Scaled values
![382651167-5c4efc93-7e6b-4118-8b15-2d839a13f6f7](https://github.com/user-attachments/assets/90677142-4acf-437b-b5d1-371fb3fa08dd)

Predicted Values
![382651288-cb702337-7080-4e6d-b449-bd8adc5be1b4](https://github.com/user-attachments/assets/1a6fced3-106e-446d-9dfd-a09927b5c965)

# Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
