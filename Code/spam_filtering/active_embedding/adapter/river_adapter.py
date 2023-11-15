class RiverMLAdapter:  # Kelas yang digunakan sebagai adapter untuk algoritma yang diimplementasi dari sklearn
    def __init__(self, model, name):
        self.model = model
        self.name = name

    def predict(self, X):
        result = []
        for i in X:
            prediction = list(self.model.predict_proba_one(i).values())
            result.append(prediction[1])

        return result

    def fit(self, X_train, y_train, X_val, y_val):
        for X, y in zip(X_train, y_train):
            self.model = self.model.learn_one(X, y)

    def update(self, input_dim, X, y, X_val, y_val):
        self.model = self.model.learn_one(X, y)
