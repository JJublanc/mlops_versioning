import pytest
import pandas as pd
from wrapper.preprocess_wrapper import add_col_prefix_ds, get_data
import numpy as np
import mock


@pytest.fixture
def data_sample():
    return pd.DataFrame({"my_target": np.linspace(0, 1, 10),
                         "wind": np.linspace(0, 1, 10),
                         "temperature": np.linspace(0, 1, 10),
                         "identification_col": np.linspace(0, 1, 10),
                         "unused_col": np.linspace(0, 1, 10),
                         "unused_col2": np.linspace(0, 1, 10),
                         })


def test_add_col_prefix_ds(data_sample):
    data = add_col_prefix_ds(data=data_sample,
                             id_cols="identification_col",
                             target_cols="my_target",
                             drop_cols=["unused_col2", "unused_col"])

    assert list(data.columns) == ["target_my_target",
                                  "feature_wind",
                                  "feature_temperature",
                                  "id_identification_col",
                                  "drop_unused_col",
                                  "drop_unused_col2"]

    data = add_col_prefix_ds(data=data_sample,
                             id_cols="identification_col")

    assert list(data.columns) == ["feature_my_target",
                                  "feature_wind",
                                  "feature_temperature",
                                  "id_identification_col",
                                  "feature_unused_col",
                                  "feature_unused_col2"]


@mock.patch('wrapper.preprocess_wrapper.pd.read_csv',
            return_value=pd.DataFrame({"target": [0, 1],
                                       "feature": [0, 1]}))
def test_get_data(mock):
    X, y = get_data("data_path",
                    "target")

    assert list(X.columns) == ["feature"]

    X, y = get_data("data_path",
                    ["target"])

    assert list(X.columns) == ["feature"]
