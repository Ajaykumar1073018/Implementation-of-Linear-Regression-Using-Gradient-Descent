# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize Parameters – Set initial values for slope m and intercept 𝑏 and choose a learning rate 𝛼
2. Compute Cost Function – Calculate the Mean Squared Error (MSE) to measure model performance.
3. Update Parameters Using Gradient Descent – Compute gradients and update m and b using the learning rate.
4. Repeat Until Convergence – Iterate until the cost function stabilizes or a maximum number of iterations is reached.

## Program:
```python
/*
Program to implement the linear regression using gradient descent.
Developed by: AJAY KUMAR .T
RegisterNumber:  212223047001
*/
import numpy as np
 import pandas as pd 
from sklearn.preprocessing import StandardScaler
 def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        
        errors=(predictions-y).reshape(-1,1)
        theta -=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
 data=pd.read_csv("/content/50_Startups.csv")
 data.head()
 X=(data.iloc[1:,:-2].values)
 X1=X.astype(float)
 scaler=StandardScaler()
 y=(data.iloc[1:,-1].values).reshape(-1,1)
 X1_Scaled=scaler.fit_transform(X1)
 Y1_Scaled=scaler.fit_transform(y)
 print(X)
 print(X1_Scaled)
 theta=linear_regression(X1_Scaled, Y1_Scaled)
 new_data= np.array([165349.2 , 136897.8 , 471784.1]).reshape(-1,1)
 new_scaled=scaler.fit_transform(new_data)
 prediction=np.dot(np.append(1, new_scaled), theta)
 prediction= prediction.reshape(-1,1)
 pre = scaler.inverse_transform(prediction)
 print(prediction)
 print(f"Predicted value: {pre}")
```

## Output:
![1ef37f1c-9030-4550-919f-369d2513fed3](https://github.com/user-attachments/assets/5cd743cc-1dfd-4aef-a643-c4c434415ee1)


![cea741a0-6de1-4b71-b444-bbc9d0e6d836](https://github.com/user-attachments/assets/1fcad0df-f296-44d4-85db-59195c524c33)
![52259f99-cbff-4241-93da-66fd2dfad124](https://github.com/user-attachments/assets/776ffe42-6601-47c5-81a0-4ed5a6dfa05c)

![d0669388-6d5f-43f4-8c2c-0462d5027ae0](https://github.com/user-attachments/assets/dc313697-e123-4b8d-80e5-83ed80cc5ab5)
![c1ef3d1d-3b67-499e-98a0-51a85447fbc0](https://github.com/user-attachments/assets/60432568-2543-43e6-a56c-de1d8789b086)

![f2884046-a970-4427-8836-77746ff353b5](https://github.com/user-attachments/assets/852e6802-a753-4e1d-b679-fceb10f71b0d)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
