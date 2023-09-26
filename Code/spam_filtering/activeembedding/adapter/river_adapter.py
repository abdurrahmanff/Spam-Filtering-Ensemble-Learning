class RiverMLAdapter:  # Kelas yang digunakan sebagai adapter untuk algoritma yang diimplementasi dari sklearn
    def __init__(self, model, name):
        self.model = model
        self.name = name

    def predict(self, X):
        result = []
        for i in X:
            result.append(self.model.predict_proba_one(i)[1])

        return result

    def fit(self, X_train, y_train, X_test, y_test):
        for X, y in zip(X_train, y_train):
            self.model = self.model.learn_one(X, y)

    def update(self, X, y):
        self.model = self.model.learn_one(X, y)
