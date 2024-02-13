import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongoarrow.api import find_pandas_all

import creds


def retrieve_data_from_MongoDB(db_name, collection_name, query, columns_to_exclude):
    cluster = MongoClient(creds.Creds.URI)
    db = cluster[db_name]
    collection = db[collection_name]
    df = find_pandas_all(collection, query)
    most_recent = (
        df.day_of_retrieval.value_counts().sort_values(ascending=False).index[0]
    )
    df = df.drop(columns=columns_to_exclude).query("day_of_retrieval == @most_recent")
    return df


def preprocess_and_split_data(df):
    processed_df = (
        df.drop(columns=["day_of_retrieval", "ad_url"])
        .dropna(subset="price")
        .query("price > 100000")
        .assign(
            price=lambda df: np.log10(df.price),
            zip_code=lambda df: pd.to_numeric(df.zip_code),
        )
        .reset_index(drop=True)
    )
    X, y = processed_df.drop(columns="price"), processed_df.price
    return X, y
