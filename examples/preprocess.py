import os
from wrapper.preprocess_wrapper import preprocess_wrapper

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

target_col = 'target'

if "iris.csv" not in os.listdir("data"):
    iris = datasets.load_iris()
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                        columns=iris['feature_names'] + ['target'])

    data.to_csv('data/iris.csv', index=False)


@preprocess_wrapper
def preprocess(X: pd.DataFrame, y: pd.Series):
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
    cwd = os.getcwd()
    data_output, ts = preprocess(wrapper_branch="preprocess",
                                 wrapper_gitwd=cwd,
                                 wrapper_data_path='data/iris.csv',
                                 wrapper_target_cols="target")

