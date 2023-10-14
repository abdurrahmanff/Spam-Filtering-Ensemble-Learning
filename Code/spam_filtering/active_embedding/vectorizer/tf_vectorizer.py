from tensorflow.keras.layers import TextVectorization


class TFVectorizer:  # pake tensorflow
    def __init__(self):
        self.vectorizer = TextVectorization(output_mode="tf_idf")
        self.vocab_size = None

    def adapt(self, X):
        self.__init__()
        self.vectorizer.adapt(X)
        self.vocab_size = self.vectorizer.vocabulary_size()

    def transform(self, X):
        tfidf_res = self.vectorizer(X).numpy()
        result = [dict(zip(self.vectorizer.get_vocabulary(), row)) for row in tfidf_res]

        return result
