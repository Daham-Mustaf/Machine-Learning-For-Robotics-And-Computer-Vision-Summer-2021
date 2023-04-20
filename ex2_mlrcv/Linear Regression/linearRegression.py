
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



class LinearRegression:
    def __init__(self, lr=0.001,n_iters=1000 ):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples)* np.dot(X.T, y_pred- Y)
            db = (1/n_samples)* np.sum(y_pred - Y)
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

df = pd.read_csv('linear.csv')
X = df['X'].values
y = df['Y'].values
# plt.scatter(X, y)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

fig = plt.figure(figsize=(8,6))
plt.scatter(X, y, color = "b", marker = "o", s = 30)
plt.show()

reg = LinearRegression(lr=0.01)
reg.fit(X,y)
predictions = reg.predict(X)

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)

mse = mse(y, predictions)
print(mse)

y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(, x, color=cmap(0.9), s=10)
m2 = plt.scatter(x, x, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()
