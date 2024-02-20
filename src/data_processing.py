from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongoarrow.api import find_pandas_all


def retrieve_data_from_MongoDB(
    db_name: str,
    collection_name: str,
    query: Dict[str, Any],
    columns_to_exclude: Optional[List[str]],
    client: Optional[MongoClient] = None,
    most_recent: bool = True,
) -> pd.DataFrame:
    """
    This function retrieves data from a MongoDB collection and returns it as a DataFrame.

    Parameters:
    - db_name (str): The name of the MongoDB database.
    - collection_name (str): The name of the collection in the database.
    - query (Dict[str, Any]): The query to use when retrieving data from the collection.
    - columns_to_exclude (List[str], optional): A list of column names to exclude from the DataFrame. If None, no columns are excluded.
    - client (MongoClient, optional): The MongoDB client to use. If None, a new client is created.
    - most_recent (bool, optional): Whether to only include the most recent data in the DataFrame. Defaults to True.

    Returns:
    - pd.DataFrame: The retrieved data as a DataFrame.
    """
    db = client[db_name]
    collection = db[collection_name]
    df = find_pandas_all(collection, query)
    if columns_to_exclude:
        df = df.drop(columns=columns_to_exclude)
    if most_recent:
        most_recent_values = str(
            pd.to_datetime(df["day_of_retrieval"].unique(), format="%Y-%m-%d")
            .sort_values(ascending=False)[0]
            .strftime("%Y-%m-%d")
        )
        df = df.query("day_of_retrieval == @most_recent_values")
        return df
    else:
        return df


def preprocess_and_split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    This function preprocesses the input DataFrame by dropping unnecessary columns,
    removing rows with missing price data, filtering out houses with prices less than 100,000,
    and transforming the price and zip code columns. It then splits the processed DataFrame
    into features (X) and target (y).

    Parameters:
    - df (pd.DataFrame): The original DataFrame to be preprocessed.

    Returns:
    - Tuple[pd.DataFrame, pd.Series]: A tuple containing the features DataFrame (X) and the target Series (y).
    """
    processed_df = (
        df.drop(columns=["day_of_retrieval", "ad_url"])
        .dropna(subset="price")
        .query("price > 100_000")
        .assign(
            price=lambda df: np.log10(df.price),
            zip_code=lambda df: pd.to_numeric(df.zip_code),
        )
        .reset_index(drop=True)
    )
    X, y = processed_df.drop(columns="price"), processed_df.price
    return X, y
