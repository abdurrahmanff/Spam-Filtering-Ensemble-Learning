import tensorflow as tf


class TensorIncrementalClassifier:
    def __init__(self):
        self.es = tf.keras.callbacks.EarlyStopping(
            monitor="precision", patience=3, mode="max", restore_best_weights=True
        )
        self.base_layers = []
        self.weights = None
        self.model: None | tf.keras.Sequential = None

    def create_model(self, input_dim):
        embedding_layer = tf.keras.layers.Embedding(
            input_dim,
            output_dim=64,
            # mask_zero=True
        )
        model = tf.keras.Sequential([embedding_layer])
        for layer in self.base_layers:
            model.add(layer)

        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        return model

    def save_weight(self):
        self.weights = self.model.get_weights()

    def update(self, input_dim, X_train, y_train, X_val, y_val):
        self.save_weight()
        self.model = self.create_model(input_dim)
        self.model.set_weights(self.weights)
        self.fit(
            X_train=X_train,
            y_train=y_train,
            X_test=X_val,
            y_test=y_val,
        )

    def fit(self, X_train, y_train, X_test, y_test):
        self.model.fit(
            X_train,
            y_train,
            epochs=50,
            validation_data=(X_test, y_test),
            # callbacks=[self.es],
        )

    def predict(self, X):
        return self.model.predict(X)
