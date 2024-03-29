from azureml.core import Workspace
from azure.storage.blob import BlobServiceClient
import mlflow
import os
import pandas as pd
import pickle
from get_data import get_data_from_storage
from ml_versioning_wrapper.commit import check_branch, commit_code


# TODO : add link to commit in mlflow params
# TODO : add link to data in mlflow params
def train_wrapper(func):
    def wrapper(wrapper_branch: str,
                wrapper_gitwd: str,
                wrapper_input_data: dict,
                wrapper_mlflow_azure: bool = False,
                wrapper_azure_container_name: str = None,
                wrapper_experiment_name: str = 'default_model',
                *args, **kwargs):
        ##########################
        # Set repo git in python #
        ##########################
        repo, repo_url = check_branch(wrapper_branch, wrapper_gitwd)

        ############
        # Get data #
        ############
        data = dict()
        for key, value in wrapper_input_data.items():
            get_data_from_storage(wrapper_azure_container_name,
                                  value)
            data[key] = pd.read_csv("./data/" + value)

        #####################
        # Set mlflow params #
        #####################
        mlflow.set_experiment(wrapper_experiment_name)

        #################
        # Config mlFlow #
        #################

        if wrapper_mlflow_azure:
            ws = Workspace.from_config()
            mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
            mlflow.set_experiment(wrapper_experiment_name)

        with mlflow.start_run():

            ####################
            # train your model #
            ####################

            results = func(*args, **data, **kwargs)

            ####################
            # log your results #
            ####################

            for object_ in results.keys():
                obj = results[object_]
                if isinstance(obj, dict):
                    mlflow.log_metrics(obj)
                else:
                    pickle.dump(obj, open(f"{object_}.pkl", 'wb'))
                    mlflow.log_artifact(f"{object_}.pkl")
                    os.remove(f"{object_}.pkl")

            mlflow.log_params(wrapper_input_data)
            mlflow.log_params({"repo url":repo_url})
            run = mlflow.active_run()
            run_id = run.info.run_id

        ###############
        # Commit code #
        ###############

        commit_code(repo, f"exp(train): run_id={run_id}")

        return results

    return wrapper
