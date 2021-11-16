import os
import pandas as pd
from wrapper.train_wrapper import train_wrapper

branch_to_exp = "train"


@train_wrapper
def train(data: pd.DataFrame):
    metrics = {"len_data": len(data)}
    model = ["object", "that", "can", "be", "pickled!"]

    results = {"metrics": metrics,
               "model": model}
    return results


if __name__ == "__main__":
    cwd = os.getcwd()
    train(wrapper_origin_file_name="X_train.csv",
          wrapper_branch="main",  # branch_to_exp,
          wrapper_gitwd=cwd,
          wrapper_mlflow_azure=True,
          wrapper_azure_container_name="data")
