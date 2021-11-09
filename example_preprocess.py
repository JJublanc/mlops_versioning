import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import logging

target_col = 'target'

if "iris.csv" not in os.listdir("data"):
    iris = datasets.load_iris()
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                        columns=iris['feature_names'] + ['target'])

    data.to_csv('data/iris.csv', index=False)


def preprocess(data: pd.DataFrame,
               target_col: str,
               features_cols: list):
    X = data[features_cols]
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.4,
                                                        random_state=42)

    X_test, X_evaluate, y_test, y_evaluate = train_test_split(X_test,
                                                              y_test,
                                                              test_size=0.4,
                                                              random_state=42)

    data_output = {"X_train": X_train,
                   "X_test": X_test,
                   "X_evaluate": X_evaluate,
                   "y_train": y_train,
                   "y_test": y_test,
                   "y_evaluate": y_evaluate}
    return data_output


if __name__ == "__main__":
    data = pd.read_csv("data/iris.csv")
    features_cols = list(data.columns.drop(target_col))
    data_output = preprocess(data,
                             target_col="target",
                             features_cols=features_cols)

    for key, value in data_output.items():
        value.to_csv(f"./data/{key}.csv", index=False)
        logging.info(f"./data/{key}.csv")
        print(f"./data/{key}.csv saved")