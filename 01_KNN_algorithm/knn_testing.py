"""_summary_
        Testing the knn model
"""
from sklearn import datasets  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from knn_algorithm import KNN

# main function
if __name__ == "__main__":
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"{X_train.shape}, {y_train.shape}")
    print(f"{X_test.shape}, {y_test.shape}")
    print(f"first row of X_train: {X_train[0]}, {type(X_train)}")
    print(f"first row of y_train: {y_train[0]}, {type(y_train)}")

    print("\n\n---- KNN ----\n")

    # create KNN classifier based on our own implementation from scratch
    classifier = KNN(k=5)
    print(f"my classifier: {classifier}")

    # train the model
    classifier.fit(X_train, y_train)

    # predict the test set
    y_pred = classifier.predict(X_test)
    print(f"y_test: {y_test}")

    # calculate the accuracy
    accuracy = classifier.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")
