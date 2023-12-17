"""_summary_

    Linear & Logistic Regression implementation with Gradient Descent
"""
import numpy as np


class BaseRegression:
    """_summary_

    Base class for Linear & Logistic Regression
    """

    def __init__(self, learning_rate=0.001, n_iters=1000):
        """_summary_
            Linear & Logistic Regression implementation with Gradient Descent

        Args:
            learning_rate (float, optional): _description_. Defaults to 0.001.
            n_iters (int, optional): _description_. Defaults to 1000.
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, train_data, train_values):
        """_summary_
            Fit the training data

        Args:
            train_data (np.ndarray): training data
            train_values (np.ndarray): target values
        """
        n_samples, n_features = train_data.shape

        self.weights, self.bias = np.zeros(n_features), 0

        # Minimizing loss, and finding the correct Weights and biases using Gradient Descent
        for _ in range(self.n_iters):
            y_predicted = self._approximation(train_data, self.weights, self.bias)

            dw = (1 / n_samples) * np.dot(train_data.T, (y_predicted - train_values))
            db = (1 / n_samples) * np.sum(y_predicted - train_values)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """_summary_
            Predict the target values

        Args:
            test_data (np.ndarray): test data

        Returns:
            _type_: predicted values/classes
        """
        return self._predict(test_data, self.weights, self.bias)

    def _predict(self, input_features, w, b):
        raise NotImplementedError

    def _approximation(self, input_features, w, b):
        raise NotImplementedError


class LinearRegression(BaseRegression):
    """_summary_
        Linear Regression implementation with Gradient Descent

    Args:
        BaseRegression (_type_): Base Regression class
    """

    def _approximation(self, input_features, w, b):
        return np.dot(input_features, w) + b

    def _predict(self, input_features, w, b):
        return np.dot(input_features, w) + b

    def mean_squared_error(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """_summary_
            Return the mean squared error


        Args:
            x_test (np.ndarray): test data
            y_test (np.ndarray): test labels

        Returns:
            float: mean squared error of the model
        """
        return np.mean((y_test - self.predict(x_test)) ** 2)

    def r2_score(self, x_test: np.ndarray, y_test: np.ndarray[np.float64]) -> float:
        """_summary_
            Return the r2 score


        Args:
            x_test (np.ndarray): test data
            y_test (np.ndarray): test labels

        Returns:
            float: r2 score of the model
        """
        y_avg = np.mean(y_test)

        model_r2_score = 1 - (
            np.sum((y_test - self.predict(x_test)) ** 2) / np.sum((y_test - y_avg) ** 2)
        )

        return model_r2_score


class LogisticRegression(BaseRegression):
    """_summary_
        Logistic Regression implementation with Gradient Descent

    Args:
        BaseRegression (_type_): Base Regression class
    """

    def _approximation(self, input_features, w, b):
        linear_model = np.dot(input_features, w) + b
        return self._sigmoid(linear_model)

    def _predict(self, input_features, w, b):
        linear_model = np.dot(input_features, w) + b
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (np.exp(-x) + 1)

    def accuracy(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """_summary_
            Return the accuracy of the model


        Args:
            x_test (np.ndarray): test data
            y_test (np.ndarray): test labels

        Returns:
            float: accuracy of the model
        """
        y_predicted = self.predict(x_test)
        return np.mean(y_predicted == y_test)
