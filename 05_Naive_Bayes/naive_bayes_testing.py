"""_summary_

    testing logistic regression algorithm
"""

from sklearn.model_selection import train_test_split  # type: ignore
from sklearn import datasets  # type: ignore
from naive_bayes_algorithm import GaussianNaiveBayes

# Testing
if __name__ == "__main__":
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    classifier = GaussianNaiveBayes()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    print(
        f"Gaussian naive bayes classification accuracy: {classifier.accuracy(X_test, y_test)}"
    )
