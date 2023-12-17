"""_summary_

    testing linear regression algorithm
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn import datasets  # type: ignore
from linear_regression_algorithm import LinearRegression

# Testing
if __name__ == "__main__":
    # Imports

    X, y, original_coef = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4, coef=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    mse = regressor.mean_squared_error(X_test, y_test)
    print("MSE:", mse)

    accu = regressor.r2_score(X_test, y_test)
    print("R2:", accu)

    # Plot
    y_pred_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.show()
