import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    This function should calculate the root mean squared error given target y and prediction y_pred

    Args:
        - y(np.array): target data
        - y_pred(np.array): predicted data

    Returns:
        - err (float): root mean squared error between y and y_pred

    """
    # Calculate the squared differences between y and y_pred
    diff_squared = (y - y_pred) ** 2
    
    # Calculate the mean of the squared differences
    mean_diff_squared = np.mean(diff_squared)
    
    # Calculate the root of the mean squared differences
    err = np.sqrt(mean_diff_squared)
    return err

def split_data(x: np.ndarray, y: np.ndarray)-> float:
    """
    This function should split the X and Y data in training, validation

    Args:
        - x: input data
        - y: target data

    Returns:
        - x_train: input data used for training
        - y_train: target data used for training
        - x_val: input data used for validation
        - y_val: target data used for validation

    """
    n = len(x)
    indices = np.random.permutation(n)
    split_idx = int(0.8 * n)

    x_train = x[indices[:split_idx]]
    y_train = y[indices[:split_idx]]
    x_val = x[indices[split_idx:]]
    y_val = y[indices[split_idx:]]

    return x_train, y_train, x_val, y_val


class LinearRegression:
    def __init__(self):
        self.theta_0 = None
        self.theta_1 = None

    def calculate_theta(self, x: np.ndarray, y: np.ndarray):
        """
        This function should calculate the parameters theta0 and theta1 for the regression line
        Args:
            - x (np.array): input data
            - y (np.array): target data

        """
        pass

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta0 and theta1 to predict the y value given an input x

        Args:
            - x: input data

        Returns:
            - y_pred: y computed w.r.t. to input x and model theta0 and theta1

        """
        
        y_pred = None

        return y_pred

class NonLinearRegression:
    def __init__(self):
        self.theta = None

    def calculate_theta(self, x: np.ndarray, y: np.ndarray, degree: Optional[int] = 2):
        """
        This function should calculate the parameters theta for the regression curve.
        In this case there should be a vector with the theta parameters (len(parameters)=degree + 1).

        Args:
            - x: input data
            - y: target data
            - degree (int): degree of the polynomial curve

        Returns:

        """
        pass

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta to predict the y value given an input x

        Args:
            - x: input data

        Returns:
            - y: y computed w.r.t. to input x and model theta parameters
        """
        
        y_pred = None

        return y_pred



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
