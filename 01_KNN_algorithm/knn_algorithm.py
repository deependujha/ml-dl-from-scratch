"""_summary_

    KNN algorithm implementation in python
"""
import numpy as np
from collections import Counter
from typing import Any


def calculate_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """_summary_
        Calculate the distance between two data points

    Args:
        x1 (np.ndarray): First data point
        x2 (np.ndarray): Second data point

    Returns:
        float: Distance between the two data points
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


# KNN algorithm


class KNN:
    """_summary_
    KNN algorithm implementation in python from scratch
    """

    def __init__(self, k=3):
        """_summary_:
            KNN algorithm implementation in python from scratch


        Args:
            k (int, optional): Number of nearest neighbors. Defaults to 3.
        """
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        """_summary_
            Fit the model

        Args:
            X (np.ndarray): X training data
            y (np.ndarray): y training labels
        """
        self.X = X
        self.y = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """_summary_
            Predict the labels for the given data


        Args:
            X (np.ndarray): X test data

        Returns:
            np.ndarray: Predicted labels for the given data
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x: np.ndarray) -> int:
        """_summary_
            Predict the label for a single data point

        Args:
            x (np.ndarray): X test data of a single data point

        Returns:
            int: Predicted label for the given data point
        """
        distances = [calculate_distance(x_train, x) for x_train in self.X]
        nearest_k_indices = np.argsort(distances)[: self.k]
        nearest_k_labels = [self.y[i] for i in nearest_k_indices]
        most_common_label = Counter(nearest_k_labels).most_common(1)[0][0]
        return most_common_label

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """_summary_
            Calculate the accuracy of the model


        Args:
            X (np.ndarray): X test data
            y (np.ndarray): y test labels

        Returns:
            float: Accuracy of the model
        """
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)

    def __repr__(self) -> str:
        """_summary_
            Return the string representation of the model


        Returns:
            str: String representation of the model
        """
        return f"KNN(k={self.k})"
