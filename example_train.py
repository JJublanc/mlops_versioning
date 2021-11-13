import os
import pandas as pd
from wrapper.train_wrapper import train_wrapper

branch_to_exp = "train"


@train_wrapper
def train(data: pd.Datframe):
    metrics = {"len_data": len(data)}
    model = ["object", "that", "can", "be", "pickled!"]

    results = {"metrics": metrics,
               "model": model}
    return results


if __name__ == "__main__":
    cwd = os.getcwd()
    train(wrapper_branch="main",  # branch_to_exp,
          wrapper_gitwd=cwd)
