"""
Dudley Spence
Single Perceptron
Assignment part 1
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
# this import allows my environment to import the iris data set using URL
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.errors = []

    def fit(self, X, y):
        """
        receives the training samples and labels
        and trains the perceptron
        """
        n_samples, n_features = X.shape

        # initiates weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for j in tqdm(range(self.n_iters)):
            errors = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.step_func(linear_output)
                update = self.lr * (y_[idx]-y_predicted)
                self.weights += update * x_i
                self.bias += update
                if update != 0:
                    errors += 1
            self.errors.append(errors)

    def predict(self, X):
        """
        receives the test data, predicts classification"
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.step_func(linear_output)
        return y_predicted

    @staticmethod
    def step_func(x):
        """
        a simple step function
        """
        return np.where(x >= 0, 1, 0)


def accuracy(y_true, y_pred):
    """
    uses the predicted and
    target to determine the
    accuracy of the perceptron classification.
    """
    print(y_pred)
    print(y_true)
    accuracyVal = np.sum(y_true == y_pred) / len(y_true)
    return accuracyVal



def choose_data_set(data_set=""):
    """
    This sets X and y to the chosen data set
    """
    if data_set == "iris":
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
        df.tail()

        X = df.iloc[50:150, [1, 3]].values
        y = df.iloc[50:150, 4].values
        # converts the labels from strings to either 1 or 0
        y = np.where(y == y[0], 1, 0)
    else:
        X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
    return X, y


if __name__ == "__main__":


    X, y = choose_data_set("iris")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


    p = Perceptron(learning_rate=0.01, n_iters=80)

    p.fit(X_train, y_train)

    predictions = p.predict(X_test)

    print("Classification accuracy", accuracy(y_test, predictions))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin-1, ymax+1])

    plt.xlabel('')
    plt.ylabel('')
    plt.title("")
    plt.show()

    plt.plot(range(1, len(p.errors) + 1), p.errors, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Number of errors')


    plt.show()