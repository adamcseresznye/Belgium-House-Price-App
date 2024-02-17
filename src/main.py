import os
import sys
from typing import List, Tuple

import joblib
import mapie
import numpy as np
import pandas as pd
import pymongo
from requests_html import HTMLSession
from sklearn import model_selection

import data_processing
import model
import preprocessing_transformers
import scraper
import utils


def connect_to_mongo(
    mongo_uri: str,
) -> Tuple[pymongo.MongoClient, pymongo.database.Database]:
    """
    This function connects to a MongoDB database and returns the client and database objects.
    If the connection fails, the function exits the program.

    Parameters:
    - mongo_uri (str): The URI of the MongoDB database.

    Returns:
    - Tuple[pymongo.MongoClient, pymongo.database.Database]: A tuple containing the client and database objects.

    Raises:
    - SystemExit: If the connection to the MongoDB server fails.
    """
    try:
        client = pymongo.MongoClient(mongo_uri)
        db = client.development
        print("Connected to MongoDB")
        return client, db
    except pymongo.errors.ServerSelectionTimeoutError:
        sys.exit("Failed to connect to MongoDB: Server not available")


def scrape_data(
    session: HTMLSession, urls: List[str], db: pymongo.database.Database, N: int = 100
) -> None:
    """
    This function scrapes data from a list of URLs using a given session, processes the data,
    and inserts it into a MongoDB database. The session is closed and a new one is created every N URLs.

    Parameters:
    - session (HTMLSession): The session used to send HTTP requests.
    - urls (List[str]): The list of URLs to scrape.
    - db (Database): The MongoDB database where the scraped data is to be inserted.
    - N (int, optional): The number of URLs to scrape before resetting the session. Defaults to 100.

    Raises:
    - Exception: If an error occurs while processing a URL.
    """
    ...
    try:
        for i, url in enumerate(urls):
            if i % N == 0:
                if session:
                    session.close()
                session = HTMLSession(browser_args=utils.Configuration.BROWSER_ARGS)
            print(f"Working on: {i} out of {len(urls)} : {url}")
            result = scraper.get_house_data(session=session, url=url)

            if result is None:
                print(f"Skipping url due to error: {url}")
                continue

            data_cleaner = scraper.DataCleaner(result)
            processed_dict = data_cleaner.process_item()
            print(processed_dict)

            db.BE_houses.insert_one(processed_dict)
    except Exception as e:
        print(f"Error while processing url {url}: {e}")


def main() -> None:
    """
    This function is the main entry point of the program. It performs the following steps:
    1. Initializes a session and retrieves URLs of houses for sale.
    2. Connects to a MongoDB database.
    3. Scrapes data from the URLs and inserts it into the database.
    4. Retrieves the scraped data from the database and preprocesses it.
    5. Splits the preprocessed data into training and test sets.
    6. Removes outliers from the training set.
    7. Creates a tuned pipeline and fits it to the training data.
    8. Evaluates the model and inserts the evaluation results into the database.
    9. Fits a MapieRegressor model to the training data and saves the model.
    10. Closes the database client and the session.

    The function does not return anything.
    """
    N = utils.Configuration.MAX_NUMBER_OF_PAGES

    session = HTMLSession(browser_args=utils.Configuration.BROWSER_ARGS)
    urls = scraper.get_house_urls(
        session=session,
        N=N,
        url="https://www.immoweb.be/en/search/house/for-sale?countries=BE&page=330&orderBy=relevance",
    )
    print("Length of urls:", len(urls))

    mongo_uri = os.getenv("MONGO_URI")
    db_client, db = connect_to_mongo(mongo_uri)

    try:
        scrape_data(session, urls, db)

        df = data_processing.retrieve_data_from_MongoDB(
            db_name="development",
            collection_name="BE_houses",
            query=None,
            columns_to_exclude="_id",
            client=db_client,
            most_recent=True,
        )
        X, y = data_processing.preprocess_and_split_data(df)

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=0.2, random_state=utils.Configuration.RANDOM_SEED
        )

        X_corr, y_corr = preprocessing_transformers.remove_outliers(X_train, y_train)

        print(f"Shape of X_train before outlier removal: {X_train.shape}")
        print(f"Shape of X_train after outlier removal: {X_corr.shape}")

        regressor = model.create_tuned_pipeline(X_corr, y_corr)

        model_evaluation = model.evaluate_model(
            regressor, X_corr, y_corr, X_test, y_test
        )
        print(model_evaluation)
        db.model_performance.insert_one(model_evaluation)

        mapie_model = mapie.regression.MapieRegressor(regressor, method="base", cv=5)
        mapie_model.fit(X_corr, y_corr)
        joblib.dump(mapie_model, utils.Configuration.MODEL.joinpath("mapie_model.pkl"))

    finally:
        db_client.close()
        session.close()


if __name__ == "__main__":
    main()
