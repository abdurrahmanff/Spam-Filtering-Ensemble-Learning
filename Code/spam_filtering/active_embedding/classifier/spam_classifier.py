class SpamClassifier:  # Kelas utama yang menampung ensemble classifier
    def __init__(self, classifiers):
        self.classifiers = (
            classifiers  # Variabel yang menampung algoritma-algoritma klasifikasi
        )

        self.prediction_threshold = (
            0.5  # Threshold dalam menentukan kelas spam (> 0.5 berarti spam)
        )
        self.evaluation = pd.DataFrame(
            columns=["Accuracy", "Precision", "Recall", "F1"]
        )  # Menampung history score tiap model sejauh iterasi

    def __train_classifiers(
        self, X_train, X_test, y_train, y_test
    ):  # Method yang digunakan untuk melakukan train semua classifiers yang menyusun ensemble classifier
        for index in range(len(self.classifiers)):
            print(
                " Train {} model ".format(self.classifiers[index].name).center(30, "+")
            )
            self.classifiers[index].fit(X_train, y_train, X_test, y_test)

    def train(self, X_train, X_test, y_train, y_test):
        self.__train_classifiers(X_train, X_test, y_train, y_test)

    def __update_classifiers(self, X, y):
        for index in range(len(self.classifiers)):
            print(
                " Update {} model ".format(self.classifiers[index].name).center(30, "+")
            )
            self.classifiers[index].update(X, y)

    def update(self, X, y):
        self.__update_classifiers(X, y)

    def __evaluate_model(
        self, y_true, prediction
    ):  # Method yang digunakan untuk mengambil evaluasi tiap model
        modelName = ""
        for classifier in self.classifiers:
            modelName += classifier.name + " "

        print(" Evaluate {} model ".format(modelName).center(30, "+"))

        accuracy = accuracy_score(y_true, prediction)
        precision = precision_score(y_true, prediction)
        recall = recall_score(y_true, prediction)
        f1 = f1_score(y_true, prediction)

        print(
            "Accuracy: {} | Precision: {} | Recall: {} | F1: {}".format(
                accuracy, precision, recall, f1
            )
        )
        curr_eval = pd.Series(
            {
                "Model": modelName,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )
        self.evaluation = pd.concat(
            [self.evaluation, curr_eval.to_frame().T], ignore_index=True
        )

    def evaluate(self, y_true, prediction):
        self.__evaluate_model(y_true, prediction)

    def __predict(self, X):  # Method yang digunakan untuk mendapatkan kelas prediksi
        raw_predictions = []
        for index in range(len(self.classifiers)):
            print(
                " Predicting using {} model ".format(
                    self.classifiers[index].name
                ).center(30, "-")
            )
            temp = self.classifiers[index].predict(X)
            temp[temp <= self.prediction_threshold] = 0
            temp[temp > self.prediction_threshold] = 1
            raw_predictions.append(temp)

        raw_predictions = np.stack((raw_predictions), axis=1)
        result = mode(raw_predictions, axis=-1, keepdims=False)
        return result.mode

    def get_prediction(self, X):
        return self.__predict(X)
