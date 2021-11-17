import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from wrapper.train_wrapper import train_wrapper

branch_to_exp = "train"


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
    data_id = str(1637122335)
    input_data = {"X_train": f"{data_id}_X_train.csv",
                  "y_train": f"{data_id}_y_train.csv",
                  "X_test": f"{data_id}_X_test.csv",
                  "y_test": f"{data_id}_y_test.csv"}

    train(wrapper_input_data=input_data,
          wrapper_branch="main",  # branch_to_exp,
          wrapper_gitwd=cwd,
          wrapper_mlflow_azure=True,
          wrapper_azure_container_name="data",
          wrapper_experiment_name="iris_classification",
          max_depth=1
          )
