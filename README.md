# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('ex3.csv')
x_raw = df['R&D Spend'].values
y_raw = df['Profit'].values

x = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

w=0.0
b = 0.0
alpha = 0.1
epochs = 100
n = len(x)

losses = []


for _ in range(epochs):
    y_hat = w * x + b

    
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    w -= alpha * dw
    b -= alpha * db
    
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, color="blue")
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("R&D Spend VS Profit")
plt.subplot(1, 2, 2)
plt.scatter(x, y, color="red", label="Data")
plt.plot(x, w * x + b, color="green", label="Regression Line")
plt.xlabel("R&D Spend (Scaled)")
plt.ylabel("Profit (Scaled)")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()

print("Final weight (w):", w)
print("Final bias (b):", b)

## Output:
<img width="1252" height="653" alt="image" src="https://github.com/user-attachments/assets/bead8210-6914-42cd-a8c9-8120fca031c9" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
