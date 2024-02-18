from unittest.mock import patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pymongo import MongoClient

from data_processing import retrieve_data_from_MongoDB


def test_retrieve_data_from_MongoDB():
    # Mocking the MongoClient
    mock_client = MongoClient()

    # Mocking the database and collection
    mock_db = mock_client["test_db"]
    mock_collection = mock_db["test_collection"]

    # Mocking the data
    mock_data = pd.DataFrame(
        {
            "column1": ["data1", "data2"],
            "column2": ["data3", "data4"],
            "day_of_retrieval": ["2024-02-17", "2024-02-18"],
        }
    )

    # Mocking the find_pandas_all function to return the mock_data
    mock_collection.find_pandas_all.return_value = mock_data

    # The expected output after dropping 'column1'
    expected_output = pd.DataFrame(
        {"column1": ["data1"], "day_of_retrieval": ["2024-02-17"]}
    )
    # The query
    query = {"day_of_retrieval": "2024-02-17"}

    # The columns to exclude
    columns_to_exclude = ["column2"]

    # Call the function with the mock client
    result = retrieve_data_from_MongoDB(
        "test_db", "test_collection", query, columns_to_exclude, mock_client, True
    )
    return mock_client


if __name__ == "__main__":
    test = test_retrieve_data_from_MongoDB()
    print(test)
