"""_summary_

    Naive Bayes Algorithm implementation from scratch
"""

import numpy as np


class GaussianNaiveBayes:
    """_summary_
    Naive Bayes Algorithm implementation from scratch
    """

    def __init__(self):
        self._prior_prob = None
        self._mean = None
        self._variance = None

    def fit(self, train_data, train_labels):
        """_summary_
            fit the model

        Args:
            train_data (_type_): train data
            train_labels (_type_): train labels
        """
        count_of_classes = np.zeros(np.unique(train_labels).shape[0])
        for _, curr_train_label in enumerate(train_labels):
            count_of_classes[curr_train_label] += 1

        self._prior_prob = count_of_classes / train_labels.shape[0]

        # now, we need a 2D array with columns indicating features and rows indicating unique labels
        # and each cell contains sum of values with [label,feature].

        # later we will use this 2D sum array to get our mean and variance
        sum_of_values = np.zeros((len(count_of_classes), train_data.shape[1]))
        self._mean = np.zeros((len(count_of_classes), train_data.shape[1]))
        self._variance = np.zeros((len(count_of_classes), train_data.shape[1]))

        for i in range(train_data.shape[0]):
            for j in range(train_data.shape[1]):
                sum_of_values[train_labels[i]][j] += train_data[i][j]

        for i in range(sum_of_values.shape[0]):
            for j in range(sum_of_values.shape[1]):
                self._mean[i][j] = sum_of_values[i][j] / count_of_classes[i]

        for i in range(train_data.shape[0]):
            for j in range(train_data.shape[1]):
                self._variance[train_labels[i]][j] += (
                    train_data[i][j] - self._mean[train_labels[i]][j]
                ) ** 2

        for i in range(self._variance.shape[0]):
            for j in range(self._variance.shape[1]):
                self._variance[i][j] /= count_of_classes[i]
        # now we need to calculate variance

    def predict(self, test_data):
        """_summary_
            test the model

        Args:
            test_data (_type_): test data
        """
        all_probabilities = np.zeros((test_data.shape[0], self._prior_prob.shape[0]))

        for i in range(test_data.shape[0]):
            for j in range(self._prior_prob.shape[0]):
                all_probabilities[i][j] = np.log(self._prior_prob[j])
                for k in range(test_data.shape[1]):
                    all_probabilities[i][j] += np.log(
                        self._gaussian_probability(
                            test_data[i][k], self._mean[j][k], self._variance[j][k]
                        )
                    )

        return np.argmax(all_probabilities, axis=1)

    def _gaussian_probability(self, x, mean, variance):
        """_summary_
            calculate gaussian probability

        Args:
            x (_type_): _description_
            mean (_type_): _description_
            variance (_type_): _description_
        """

        const_part = 1 / np.sqrt(2 * np.pi * variance)

        expo_part = np.exp(-((x - mean) ** 2) / (2 * variance))

        eta = 1e-10
        const_part = np.maximum(const_part, eta)
        expo_part = np.maximum(expo_part, eta)

        return const_part * expo_part

    def accuracy(self, test_data, test_labels):
        """_summary_
            calculate accuracy

        Args:
            test_data (_type_): test data
            test_labels (_type_): test labels
        """
        predictions = self.predict(test_data)
        return np.sum(predictions == test_labels) / test_labels.shape[0]
