import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongoarrow.api import find_pandas_all

import creds


def retrieve_data_from_MongoDB(query):
    cluster = MongoClient(creds.Creds.URI)
    df = find_pandas_all(cluster.dev.BE_houses, query)
    return df


def preprocess_and_split_data(df):
    processed_df = (
        df.drop(columns=["_id", "day_of_retrieval", "ad_url"])
        .dropna(subset="price")
        .query("price > 100000")
        .assign(price=lambda df: np.log10(df.price))
        .reset_index(drop=True)
    )
    X, y = processed_df.drop(columns="price"), processed_df.price
    return X, y
