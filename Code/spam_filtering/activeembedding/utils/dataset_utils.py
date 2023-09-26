import pandas as pd
import string
import nltk

nltk.download("punkt")
nltk.download("stopwords")
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class DatasetUtils:
    def __init__(self, dataset: pd.DataFrame, name: str):
        self.dataset = dataset
        self.name = name

    def get_column_values(self, col_name: str) -> list:
        return self.dataset[col_name].values

    def _remove_non_printable_char(self, text):
        printable = set(string.printable)
        return "".join(filter(lambda x: x in printable, text))

    def _remove_punctuation(self, text):
        return text.translate(str.maketrans("", "", string.punctuation))

    def _remove_number(self, text):
        return text.translate(str.maketrans("", "", string.digits))

    def _remove_stopword(self, tokens):
        stop_words = set(stopwords.words("english"))
        clean_tokens = []
        for token in tokens:
            if token not in stop_words:
                clean_tokens.append(token)
        return clean_tokens

    def _stem(self, tokens):
        stemmer = EnglishStemmer()
        stemmed = [stemmer.stem(token) for token in tokens]
        return stemmed

    def _remove_new_line_character(self, tokens):
        res = []
        for token in tokens:
            res.append(token.replace("\r\n", ""))
        return res

    def _preprocess_each_row(self, text):
        text = self._remove_non_printable_char(text)
        text = self._remove_number(text)
        text = text.lower()
        text = self._remove_punctuation(text)
        tokens = word_tokenize(text)
        tokens = self._remove_stopword(tokens)
        tokens = self._stem(tokens)
        result = " ".join(tokens).strip()
        return result

    def preprocess(self):
        self.dataset["preprocessed"] = self.dataset["text"].apply(
            self._preprocess_each_row
        )
        self.dataset.dropna(inplace=True)
        print("Finished preprocessing {} data".format(self.name).center(30, "-"))
