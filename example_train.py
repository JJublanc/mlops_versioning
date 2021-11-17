import os
import pandas as pd
from wrapper.train_wrapper import train_wrapper

branch_to_exp = "train"


@train_wrapper
def train(X_train: pd.DataFrame,
          y_train: pd.DataFrame,
          X_test: pd.DataFrame,
          y_test: pd.DataFrame):
    metrics = {"len_data": len(data)}
    model = ["object", "that", "can", "be", "pickled!"]

    results = {"metrics": metrics,
               "model": model}
    return results


if __name__ == "__main__":
    cwd = os.getcwd()

    input_data = {"X_train": ,
                  "y_train": ,
                  "X_test": ,
                  "y_test": }

    train(wrapper_input_data={"X_train":,
          wrapper_branch="main",  # branch_to_exp,
          wrapper_gitwd=cwd,
          wrapper_mlflow_azure=True,
          wrapper_azure_container_name="data")
