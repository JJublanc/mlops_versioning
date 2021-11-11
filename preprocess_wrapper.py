from wrapper_tools.commit import check_branch, commit_code
import pandas as pd
from typing import Union

def get_data(data_path: str,
             target_cols: Union[str, list]):
    data = pd.read_csv(data_path)

    if isinstance(target_cols, str):
        other_columns = list(set(data.columns) - set([target_cols]))
    elif isinstance(target_cols, list):
        other_columns = list(set(data.columns) - set(target_cols))
    X = data[other_columns]
    y = data[target_cols]

    return X, y


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
    def wrapper(wrapper_branch: str,
                wrapper_gitwd: str,
                wrapper_data_path: str,
                wrapper_id_cols: Union[list, str],
                wrapper_drop_cols: Union[list, str],
                wrapper_target_cols: Union[list, str]):

        ##########################
        # Set repo git in python #
        ##########################
        repo = check_branch(wrapper_branch, wrapper_gitwd)

        ############
        # Get data #
        ############

        X, y = get_data(path=wrapper_data_path,
                        target_cols=wrapper_target_cols)
        data = pd.read_csv(wrapper_data_path)
        other_columns = list(set(data.columns) - set(wrapper_target_cols))
        X = data[other_columns]
        y = data[wrapper_target_cols]

        ##############
        # Preprocess #
        ##############

        output = func(X, y)

        ##################
        # add col prefix #
        ##################

        if isinstance(output, dict):
            for key, data in output.items():
                if isinstance(data, pd.DataFrame):
                    output[key] = add_col_prefix_ds(data,
                                                    wrapper_id_cols,
                                                    wrapper_target_cols,
                                                    wrapper_drop_cols)
        elif isinstance(output, pd.DataFrame):
            output = add_col_prefix_ds(output,
                                       wrapper_id_cols,
                                       wrapper_target_cols,
                                       wrapper_drop_cols)

        ##############################
        # TODO: Create global prefix #
        ##############################

        ###################
        # TODO: Save data #
        ###################

        return output

    return wrapper
