from river.feature_extraction import TFIDF


class RiverVectorizer:  # ini yang pake river
    def __init__(self):
        self.vectorizer = TFIDF()
        self.vocab_size = None

    def adapt(self, X):
        for sentence in X:
            self.vectorizer = self.vectorizer.learn_one(sentence)
        self.vocab_size = len(self.vectorizer.dfs)

    def transform(self, X):
        result = []
        for row in X:
            raw_vector = self.vectorizer.transform_one(row)
            full_vector = {}
            for word in self.vectorizer.dfs.keys():  # cek semua vocab
                value = 0
                if word in raw_vector.keys():
                    value = raw_vector[word]
                full_vector[word] = value
            result.append(full_vector)

        return result
