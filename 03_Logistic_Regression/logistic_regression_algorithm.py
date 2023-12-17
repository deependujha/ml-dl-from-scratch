"""_summary_

    Logistic Regression implementation with Gradient Descent
"""
import numpy as np


class LogisticRegression:
    """_summary_

    Logistic Regression implementation with Gradient Descent
    """

    def __init__(self, learning_rate=0.001, n_iters=1000):
        """_summary_
            Logistic Regression implementation with Gradient Descent

        Args:
            learning_rate (float, optional): _description_. Defaults to 0.001.
            n_iters (int, optional): _description_. Defaults to 1000.
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, train_data: np.ndarray, train_labels: np.ndarray):
        """_summary_
            Fit the training data

        Args:
            x (np.ndarray): training data
            y (np.ndarray): target values
        """
        n_samples, n_features = train_data.shape

        # init parameters
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

        # gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(train_data, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(train_data.T, (y_predicted - train_labels))
            db = (1 / n_samples) * np.sum(y_predicted - train_labels)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """_summary_
            Predict the target values

        Args:
            test_data (np.ndarray): test data

        Returns:
            _type_: _description_
        """
        linear_model = np.dot(test_data, self.weights) + self.bias
        y_approximated = [1 if self._sigmoid(i) > 0.5 else 0 for i in linear_model]
        return np.array(y_approximated)

    @staticmethod
    def _sigmoid(x: float) -> float:
        """_summary_
            Return the sigmoid of the input

        Args:
            x (np.ndarray): input

        Returns:
            np.ndarray: sigmoid of the input
        """
        return 1 / (1 + np.exp(-x))

    def coef_(self) -> np.ndarray[np.float64]:
        """_summary_
            Return the weights

        Returns:
            np.ndarray: weights of the model
        """
        if self.weights is None:
            raise ValueError("Model not fitted")
        return self.weights

    def intercept_(self) -> float:
        """_summary_
            Return the bias

        Returns:
            float: bias of the model
        """
        if self.bias is None:
            raise ValueError("Model not fitted")
        return self.bias

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

    def __repr__(self) -> str:
        """_summary_
            Return a string representation of the model

        Returns:
            str: a string representation of the model
        """
        return f"LogisticRegression(lr={self.lr}, n_iters={self.n_iters})"
