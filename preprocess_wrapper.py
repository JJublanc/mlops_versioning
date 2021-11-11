from wrapper_tools.commit import check_branch, commit_code
import pandas as pd
from typing import Union
import os
import mlflow
import pickle


def add_col_prefix_ds(data: pd.DataFrame,
                      id_cols: Union[list, str] = "",
                      target_cols: Union[list, str] = "",
                      drop_cols: Union[list, str] = ""):
    renamed_cols = {}
    for i, cols in enumerate(data.columns):
        if cols in id_cols:
            renamed_cols[data.columns[i]] = "id_" + data.columns[i]
        elif cols in drop_cols:
            renamed_cols[data.columns[i]] = "drop_" + data.columns[i]
        elif cols in target_cols:
            renamed_cols[data.columns[i]] = "target_" + data.columns[i]
        else:
            renamed_cols[data.columns[i]] = "feature_" + data.columns[i]

    data = data.rename(columns=renamed_cols)

    return data


def preprocess_wrapper(func):
    def wrapper(branch: str,
                gitwd: str,
                target_col: str,
                data_path: str,
                id_cols: list,
                drop_cols: list,
                target_cols,
                *args, **kwargs):

        ##########################
        # Set repo git in python #
        ##########################
        repo = check_branch(branch, gitwd)

        #############
        # Load data #
        #############

        data = pd.read_csv(data_path)

        ##############
        # Preprocess #
        ##############

        output = func(*args, **kwargs)

        ##################
        # add col prefix #
        ##################

        if isinstance(output, dict):
            for key, data in output.items():
                if isinstance(data, pd.DataFrame):
                    output[key] = add_col_prefix_ds(data,
                                                    id_cols,
                                                    target_cols,
                                                    drop_cols)
        elif isinstance(output, pd.DataFrame):
            output = add_col_prefix_ds(output,
                                       id_cols,
                                       target_cols,
                                       drop_cols)

        ########################
        # create global prefix #
        ########################


