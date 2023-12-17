"""_summary_

    testing logistic regression algorithm
"""

from sklearn.model_selection import train_test_split  # type: ignore
from sklearn import datasets  # type: ignore
from logistic_regression_algorithm import LogisticRegression

# Testing
if __name__ == "__main__":
    # Imports

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)
    print(
        f"Logistic regression classification accuracy: {regressor.accuracy(X_test, y_test)}"
    )
    coef_ = regressor.coef_()
    bias = regressor.intercept_()
    print(f"Coefficients : {coef_}")
    print(f"Bias: {bias}")
