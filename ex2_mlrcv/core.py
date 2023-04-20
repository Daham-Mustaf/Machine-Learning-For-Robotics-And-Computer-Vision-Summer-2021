import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_regression(x, y, y_pred):
    assert len(x.shape) == 1, f"y shape should be ({x.shape[0]},) but it is {x.shape}"
    assert len(y.shape) == 1, f"y shape should be ({y.shape[0]},) but it is {y.shape}"
    assert len(y_pred.shape) == 1, f"y_pred shape should be ({y_pred.shape[0]},) but it is {y_pred.shape}"

    plt.scatter(x, y)
    x_y = np.concatenate((x[:,np.newaxis], y_pred[:,np.newaxis]), axis=-1)
    x_y = x_y[x_y[:,0].argsort()]

    plt.plot(x_y[:,0], x_y[:,1], color='r')
    plt.show()
# Dataset read
import seaborn as sn
df = pd.read_csv('linear.csv')
x = df['X'].values
y = df['Y'].values
# X = np.hstack((np.ones((len(x), 1)), x))
# X = np.hstack((np.ones((len(x), 1)), x))
# y = y.reshape(-1, 1)
def calculate_theta( x: np.ndarray, y: np.ndarray):
    """
    This function should calculate the parameters theta0 and theta1 for the regression line

    Args:
        - x (np.array): input data
        - y (np.array): target data

    """
    # add a column of ones to the input data to account for the intercept term
    X = np.hstack((np.ones((len(x), 1)), x))
    y = y.reshape(-1, 1)
    
    # calculate theta using the normal equation
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
t = calculate_theta(x,y)