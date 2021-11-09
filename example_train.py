from train_wrapper import train_wrapper
import os

branch_to_exp = "train"

@train_wrapper
def train():
    metrics = {"accuracy": 0.6}
    model = ["object", "that", "can", "be", "pickled!"]

    results = {"metrics":metrics,
               "model":model}
    return results


if __name__ == "__main__":
    cwd = os.getcwd()
    train(branch="main", # branch_to_exp,
          gitwd=cwd)
