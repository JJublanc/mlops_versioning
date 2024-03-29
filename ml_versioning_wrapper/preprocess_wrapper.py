from azure.storage.blob import BlobServiceClient
import calendar
import logging
import os
import pandas as pd
import time
from typing import Union
from ml_versioning_wrapper.commit import check_branch, commit_code
from ml_versioning_wrapper.preprocess import get_xy_from_data_path, add_col_prefix_ds


def preprocess_wrapper(func):
    def wrapper(wrapper_branch: str,
                wrapper_gitwd: str,
                wrapper_data_path: str = "",
                wrapper_id_cols: Union[list, str] = "",
                wrapper_drop_cols: Union[list, str] = "",
                wrapper_target_cols: Union[list, str] = "",
                wrapper_azure_container_name: str = None):

        ##########################
        # Set repo git in python #
        ##########################
        repo, _ = check_branch(wrapper_branch, wrapper_gitwd)

        ############
        # Get data #
        ############

        X, y = get_xy_from_data_path(data_path=wrapper_data_path,
                                     target_cols=wrapper_target_cols)

        ##############
        # Preprocess #
        ##############

        data_output = func(X, y)

        ##################
        # add col prefix #
        ##################

        if isinstance(data_output, dict):
            for key, data in data_output.items():
                if isinstance(data, pd.DataFrame):
                    data_output[key] = add_col_prefix_ds(data,
                                                         wrapper_id_cols,
                                                         wrapper_target_cols,
                                                         wrapper_drop_cols)
        elif isinstance(data_output, pd.DataFrame):
            data_output = add_col_prefix_ds(data_output,
                                            wrapper_id_cols,
                                            wrapper_target_cols,
                                            wrapper_drop_cols)

        ########################
        # Create global prefix #
        ########################

        # get current timestamp
        gmt = time.gmtime()
        ts = calendar.timegm(gmt)

        #############
        # Save data #
        #############

        for key, value in data_output.items():
            value.to_csv(f"./data/{ts}_{key}.csv", index=False)
            logging.info(f"./data/{ts}_{key}.csv saved")

        if wrapper_azure_container_name is not None:
            for key in data_output.keys():
                # Set connection to AZURE
                connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
                blob_service_client = BlobServiceClient.from_connection_string(
                    connect_str)

                local_path = "./data"
                local_file_name = f"{ts}_{key}.csv"
                upload_file_path = os.path.join(local_path, local_file_name)

                blob_client = blob_service_client.get_blob_client(
                    container=wrapper_azure_container_name,
                    blob=local_file_name)

                # Upload the created file
                with open(upload_file_path, "rb") as data:
                    blob_client.upload_blob(data)

        ##########
        # Commit #
        ##########

        commit_code(repo, f"exp(preprocess): timestamp={ts}")

        return data_output, ts

    return wrapper
