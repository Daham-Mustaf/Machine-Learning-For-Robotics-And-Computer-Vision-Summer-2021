
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
from matplotlib import pyplot as plt

import numpy as np
def euclidian_distance(x1, x2):
    """
    Calculates the Euclidean distance between two vectors x1 and x2.

    Args:
        - x1 (numpy.ndarray): The first vector.
        - x2 (numpy.ndarray): The second vector.

    Returns:
        - dist (float): The Euclidean distance between x1 and x2.
    """
    dist = np.sqrt(np.sum((x2 - x1) ** 2))
    return dist


class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y 
        
    def predict(self, X):
        predictions= [self._predict(x)for x in X]
        return predictions

    def _predict(self, x):
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]



