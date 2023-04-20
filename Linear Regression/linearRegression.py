
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
from matplotlib import pyplot as plt

import numpy as np


class LinearRegression:

    def __init__(self, lr = 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # if X.ndim == 1:
        #     X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_samples)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db


    def predict(self, X):
        y_pred = np.dot(X, self.weights.reshape(-1, 1)) + self.bias
        return y_pred

df = pd.read_csv('linear.csv')
X = df[['X']].values
y = df['Y'].values
X_train, X_test, y_train, y_test = train_test_split(df['X'], df['Y'], test_size=0.2)
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train, y_train)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('linear.csv')
X = df[['X']]  # Use double brackets to create a 2D array
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)

clf.score(X_test, y_test)








# plt.scatter(X, y)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()
X = X.reshape(-1, 1)

n_samples, n_features = X.shape
weights = np.zeros(n_samples)

fig = plt.figure(figsize=(8,6))
plt.scatter(X, y, color = "b", marker = "o", s = 30)
plt.show()


reg = LinearRegression(lr=0.01)
reg.fit(X_train,y_train)
X_test.shape
predictions = reg.predict(X_test)

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)

mse = mse(y_test, predictions)
print(mse)
