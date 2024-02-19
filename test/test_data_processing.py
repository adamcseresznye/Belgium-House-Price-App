from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from data_processing import preprocess_and_split_data


def test_preprocess_and_split_data():
    # Create a fixture DataFrame
    df = pd.DataFrame(
        {
            "day_of_retrieval": ["2022-01-01", "2022-01-02", "2022-01-03"],
            "ad_url": ["url1", "url2", "url3"],
            "price": [100_001, 200_000, np.nan],
            "zip_code": ["12345", "23456", "34567"],
        }
    )

    # Call the function with the fixture DataFrame
    X, y = preprocess_and_split_data(df)

    # Check that the unnecessary columns have been dropped
    assert "day_of_retrieval" not in X.columns
    assert "ad_url" not in X.columns

    # Check that rows with missing price data have been removed
    assert X.shape[0] == 2

    # Check that houses with prices less than 100,000 have been filtered out
    assert (np.power(10, y) > 100_000).all()

    # Check that the price and zip code columns have been transformed
    assert (y == np.log10(df.price.dropna())).all()
    assert (X.zip_code == pd.to_numeric(df.loc[X.index, "zip_code"])).all()
