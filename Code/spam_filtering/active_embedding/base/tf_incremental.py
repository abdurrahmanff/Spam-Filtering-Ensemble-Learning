from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding
from tensorflow.keras import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall


class TensorIncrementalClassifier:
    def __init__(self):
        self.es = EarlyStopping(
            monitor="precision", patience=3, mode="max", restore_best_weights=True
        )
        self.base_layers = []
        self.weights = None
        self.model: None | Sequential = None

    def create_model(self, input_dim):
        embedding_layer = Embedding(
            input_dim,
            output_dim=64,
            # mask_zero=True
        )
        model = Sequential([embedding_layer])
        for layer in self.base_layers:
            model.add(layer)

        model.compile(
            loss=BinaryCrossentropy(),
            optimizer=Adam(),
            metrics=[
                "accuracy",
                Precision(name="precision"),
                Recall(name="recall"),
            ],
        )

        return model

    def save_weight(self):
        self.weights = self.model.get_weights()

    def update(self, input_dim, X_train, y_train):
        self.save_weight()
        self.model = self.create_model(input_dim)
        self.model.set_weights(self.weights)
        self.model.fit(X_train, y_train, epochs=50, callbacks=[self.es])

    def fit(self, X_train, y_train, X_test, y_test):
        self.model.fit(
            X_train,
            y_train,
            epochs=50,
            validation_data=(X_test, y_test),
            callbacks=[self.es],
        )

    def predict(self, X):
        return self.model.predict(X)
