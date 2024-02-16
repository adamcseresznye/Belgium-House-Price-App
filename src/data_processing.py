import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongoarrow.api import find_pandas_all

import creds


def retrieve_data_from_MongoDB(
    db_name,
    collection_name,
    query,
    columns_to_exclude,
    cluster=None,
    most_recent=True,
):
    if cluster is None:
        cluster = MongoClient(creds.Creds.URI)
    try:
        db = cluster[db_name]
        collection = db[collection_name]
        df = find_pandas_all(collection, query)
        if columns_to_exclude:
            df = df.drop(columns=columns_to_exclude)
        if most_recent:
            most_recent_values = (
                df.day_of_retrieval.value_counts().sort_values(ascending=False).index[0]
            )
            df = df.query("day_of_retrieval == @most_recent_values")
            return df
        else:
            return df
    finally:
        cluster.close()


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
