import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from ml_versioning_wrapper.train_wrapper import train_wrapper

branch_to_exp = "train"


def get_max_data_id(data_folder="./data"):
    data_ids = []
    files = os.listdir(data_folder)
    for file in files:
        try:
            id_data = int(file.split("_")[0])
            data_ids.append(id_data)
        except:
            pass
    return str(np.max(data_ids))


@train_wrapper
def train(X_train: pd.DataFrame,
          y_train: pd.DataFrame,
          X_test: pd.DataFrame,
          y_test: pd.DataFrame,
          max_depth: int):
    # Train
    clf = RandomForestClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)

    # Test
    metrics = dict()
    y_pred = clf.predict(X_test)
    metrics["accuracy"] = accuracy_score(y_test, y_pred)

    results = {"metrics": metrics,
               "model": clf}
    return results


if __name__ == "__main__":
    cwd = os.getcwd()
    data_id = get_max_data_id()
    input_data = {"X_train": f"{data_id}_X_train.csv",
                  "y_train": f"{data_id}_y_train.csv",
                  "X_test": f"{data_id}_X_test.csv",
                  "y_test": f"{data_id}_y_test.csv"}

    train(wrapper_input_data=input_data,
          wrapper_branch="train_azure",  
          wrapper_gitwd=cwd,
          wrapper_mlflow_azure=True,
          wrapper_azure_container_name="data",
          wrapper_experiment_name="iris_classification",
          max_depth=1
          )
