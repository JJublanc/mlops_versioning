import pandas as pd
from typing import Union


def get_xy_from_data_path(data_path: str,
                          target_cols: Union[str, list]) -> \
        (pd.DataFrame, Union[pd.Series, pd.DataFrame]):
    """
    From a csv filepath split data along columns to isolate target columns from
    other
    :param data_path: path to a csv file
    :param target_cols: columns name or list of names corresponding to targets
    :return: X: dataframes without target cols, y: dataframes with only target
    cols
    """
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
    """
    Add prefix to column names corresponding to its function : feature, id,
    target, unused
    :param data: DataFrame for which we want to rename columns
    :param id_cols: column or list of columns identifying rows
    :param target_cols: column or list of columns that are the target of
    the training task
    :param drop_cols: columns unused
    :return: DataFrame with renamed columns
    """
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

