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