from azureml.core import Workspace
from azure.storage.blob import BlobServiceClient
import mlflow
import os
import pandas as pd
import pickle
from wrapper.commit import check_branch, commit_code

# TODO : add link to commit in mlflow params
# TODO : add link to data in mlflow params
def train_wrapper(func):
    def wrapper(wrapper_branch: str,
                wrapper_gitwd: str,
                wrapper_origin_file_name: str,
                wrapper_mlflow_azure: bool = False,
                wrapper_azure_container_name: str = None,
                *args, **kwargs):
        ##########################
        # Set repo git in python #
        ##########################
        repo = check_branch(wrapper_branch, wrapper_gitwd)

        ############
        # Get data #
        ############

        if wrapper_azure_container_name:
            if wrapper_origin_file_name not in os.listdir("./data/"):
                connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
                blob_service_client = BlobServiceClient.from_connection_string(
                                                                   connect_str)
                blob_client = blob_service_client.get_blob_client(
                    container=wrapper_azure_container_name,
                    blob=wrapper_origin_file_name)

                # Download csv file
                with open("./data/" + wrapper_origin_file_name, "wb") as \
                        download_file:
                    download_file.write(blob_client.download_blob().readall())

        data = pd.read_csv("./data/" + wrapper_origin_file_name)

        #####################
        # Set mlflow params #
        #####################
        experiment_name = 'default_model'
        mlflow.set_experiment(experiment_name)

        #################
        # Config mlFlow #
        #################

        if wrapper_mlflow_azure:
            ws = Workspace.from_config()
            mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
            mlflow.set_experiment(experiment_name)

        with mlflow.start_run():

            ####################
            # train your model #
            ####################

            results = func(data, *args, **kwargs)

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

            run = mlflow.active_run()
            run_id = run.info.run_id

        ###############
        # Commit code #
        ###############

        commit_code(repo, f"exp(train): run_id={run_id}")

        return results

    return wrapper
