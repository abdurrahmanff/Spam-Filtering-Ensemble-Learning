from river.feature_extraction import TFIDF
import math


class RiverVectorizer:  # ini yang pake river
    def __init__(self):
        self.vectorizer = TFIDF()
        self.vocab_size = None

    def __calculate_idf(self, count):
        return math.log((1 + self.vectorizer.n) / (1 + count)) + 1

    def adapt(self, X):
        for sentence in X:
            self.vectorizer = self.vectorizer.learn_one(sentence)
        self.vectorizer.dfs = dict(
            filter(
                lambda pair: self.__calculate_idf(pair[1]) < 8,
                self.vectorizer.dfs.items(),
            )
        )
        # print(self.vectorizer.dfs)
        self.vocab_size = len(self.vectorizer.dfs)
        # print(self.vocab_size)

    def transform(self, X):
        result = []
        for row in X:
            if row not in self.vectorizer.dfs:
                pass
            raw_vector = self.vectorizer.transform_one(row)
            full_vector = {}
            for word in self.vectorizer.dfs.keys():  # cek semua vocab
                value = 0
                if word in raw_vector.keys():
                    value = raw_vector[word]
                full_vector[word] = value
            result.append(full_vector)

        return result
