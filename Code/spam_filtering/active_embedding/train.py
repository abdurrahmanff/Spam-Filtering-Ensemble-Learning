import pandas as pd
import os
import sys
import numpy as np

WORK_DIR = os.getcwd()
TRAIN_BATCH_SIZE = 5
sys.path.append(WORK_DIR + "/Code/spam_filtering/")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from active_embedding.utils import *
from active_embedding.vectorizer import RiverVectorizer as Vectorizer
from active_embedding.classifier import SpamClassifier
from active_embedding.adapter import RiverMLAdapter, TensorFlowAdapter
from active_embedding.base import TensorIncrementalClassifier
import tensorflow as tf
from river.tree import ExtremelyFastDecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

RESULT_DIR = WORK_DIR + "/Result/EnsembleLearning/tfidf/1/"


class LSTMIncremental(TensorIncrementalClassifier):
    def __init__(self, input_dim):
        super().__init__()
        self.base_layers = [
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
        self.model = self.create_model(input_dim)

    def create_model(self, input_dim):
        return super().create_model(input_dim)


class DNNIncremental(TensorIncrementalClassifier):
    def __init__(self, input_dim):
        super().__init__()
        self.base_layers = [
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
        self.model = self.create_model(input_dim)

    def create_model(self, input_dim):
        return super().create_model(input_dim)


def plot_confusion_matrix(y_true, prediction, i):
    cm = confusion_matrix(y_true, prediction)
    directory = RESULT_DIR + "Evaluation/Confusion_matrix/Train/"
    os.makedirs(directory, exist_ok=True)
    ConfusionMatrixDisplay(cm, display_labels=["false", "true"]).plot()
    plt.title("Confusion Matrix")
    plt.savefig(directory + "train_batch{}.png".format(i))


data_train_path = WORK_DIR + "/Datasets/EnronSpam/parsed/split/train.csv"
data_test_path = WORK_DIR + "/Datasets/EnronSpam/parsed/split/test.csv"
data_val_path = WORK_DIR + "/Datasets/EnronSpam/parsed/split/val.csv"

data_train = pd.read_csv(data_train_path)
data_val = pd.read_csv(data_val_path)

print(" Preprocessing Dataset ".center(30, "#"))
data_train = DatasetUtils(data_train, "Data Train")
data_val = DatasetUtils(data_val, "Data Validation")
data_train.preprocess()
data_val.preprocess()

X_train_all = data_train.get_column_values("preprocessed")
X_val = data_val.get_column_values("preprocessed")
y_train_all = data_train.get_column_values("label")
y_val = data_val.get_column_values("label")

text_vectorizer = Vectorizer()
text_vectorizer.adapt(X_train_all)

print(" Vectorizing Validation Dataset ".center(30, "#"))
X_val = text_vectorizer.transform(X_val)

X_train_batches = np.array_split(X_train_all, TRAIN_BATCH_SIZE)
y_train_batches = np.array_split(y_train_all, TRAIN_BATCH_SIZE)


classifiers = [
    TensorFlowAdapter(LSTMIncremental(text_vectorizer.vocab_size), "LSTM"),
    TensorFlowAdapter(DNNIncremental(text_vectorizer.vocab_size), "DNN"),
    RiverMLAdapter(ExtremelyFastDecisionTreeClassifier(), "Decision Tree"),
]

spam_classifier = SpamClassifier(classifiers)

for i in range(0, TRAIN_BATCH_SIZE):
    print(" Batch {} ".format(i + 1).center(30, "-"))

    print(" Vectorizing Training Dataset ".center(30, "#"))
    X_train = text_vectorizer.transform(X_train_batches[i])

    print(" Train data ".center(40, "#"))
    spam_classifier.update(
        text_vectorizer.vocab_size, X_train, y_train_batches[i], X_val, y_val
    )

    print(" Evaluate models ".center(30, "#"))
    prediction = spam_classifier.get_prediction(X_val)
    spam_classifier.evaluate(y_val, prediction)

    plot_confusion_matrix(y_val, prediction, i)
