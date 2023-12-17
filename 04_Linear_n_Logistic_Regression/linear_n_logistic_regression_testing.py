"""_summary_

    testing linear & logistic regression algorithm
"""

from sklearn.model_selection import train_test_split  # type: ignore
from sklearn import datasets  # type: ignore
from linear_n_logistic_regression_algorithm import LinearRegression, LogisticRegression


if __name__ == "__main__":
    # Linear Regression
    X, y, _ = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4, coef=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    mse = regressor.mean_squared_error(X_test, y_test)
    print("Linear reg mean-squared-error:", mse)

    accu = regressor.r2_score(X_test, y_test)
    print("Linear reg R2 score:", accu)

    print("--------------------------------------------------------")

    # Logistic reg
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    classifier = LogisticRegression(learning_rate=0.0001, n_iters=1000)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    print("Logistic reg classification accuracy:", classifier.accuracy(X_test, y_test))
