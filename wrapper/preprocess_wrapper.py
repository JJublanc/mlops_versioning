import calendar
import time

from wrapper.commit import check_branch, commit_code
from wrapper.preprocess import get_data, add_col_prefix_ds
import pandas as pd
from typing import Union
import logging


def preprocess_wrapper(func):
    def wrapper(wrapper_branch: str,
                wrapper_gitwd: str,
                wrapper_data_path: str = "",
                wrapper_id_cols: Union[list, str] = "",
                wrapper_drop_cols: Union[list, str] = "",
                wrapper_target_cols: Union[list, str] = ""):

        ##########################
        # Set repo git in python #
        ##########################
        repo = check_branch(wrapper_branch, wrapper_gitwd)

        ############
        # Get data #
        ############

        X, y = get_data(data_path=wrapper_data_path,
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

        ##########
        # Commit #
        ##########

        commit_code(repo, f"exp(preprocess): timestamp={ts}")

        return data_output

    return wrapper
