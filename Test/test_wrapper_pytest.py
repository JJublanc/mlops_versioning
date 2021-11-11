import pytest
import pandas as pd
from preprocess_wrapper import add_col_prefix_ds
import numpy as np


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
