from active_embedding.base import TensorIncrementalClassifier
import numpy as np


class TensorFlowAdapter:  # Kelas yang digunakan sebagai adapter untuk algoritma yang diimplementasi dari tensorflow
    def __init__(self, model: TensorIncrementalClassifier, name):
        self.model = model
        self.name = name

    def predict(self, X):
        X = self.__extract_value(X)
        return self.model.predict(X).flatten()

    def __extract_value(self, X):
        return np.array([list(row.values()) for row in X])

    def fit(self, X_train, y_train, X_test, y_test):
        X_train = self.__extract_value(X_train)
        X_test = self.__extract_value(X_test)
        self.model.fit(X_train, y_train, X_test, y_test)

    def update(self, input_dim, X_train, y_train, X_val, y_val):
        X_train = self.__extract_value(X_train)
        self.model.update(input_dim, X_train, y_train, X_val, y_val)
