from sklearn.datasets import load_files
import numpy as np
import os
import pandas as pd

for i in range(1, 7):
    emails = load_files(
        os.getcwd() + "\\Datasets\\EnronSpam\\enron{}".format(i),
        random_state=42,
        encoding="ISO-8859-1",
    )
    X = np.array(emails.data)
    y = np.array(emails.target)
    pd.DataFrame({"email": X, "label": y}).to_csv(
        "EnronSpam{}.csv".format(i), index=False, escapechar="\\"
    )
