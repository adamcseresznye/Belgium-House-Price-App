import os
import sys
from datetime import date
from io import StringIO

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


def main():
    N = utils.Configuration.MAX_NUMBER_OF_PAGES

    session = HTMLSession(browser_args=utils.Configuration.BROWSER_ARGS)
    # get links to houses
    urls = scraper.get_house_urls(
        session=session,
        N=N,
    )
    print("Length of urls:", len(urls))

    # connect to mongodb
    mongo_uri = os.getenv("MONGO_URI")
    client = pymongo.MongoClient(mongo_uri)
    db = client.development
    if client:
        print("Connected to MongoDB")
    else:
        sys.exit("Failed to connect to MongoDB")
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

        # get data and preprocess
        df = data_processing.retrieve_data_from_MongoDB(
            db_name="development",
            collection_name="BE_houses",
            query=None,
            columns_to_exclude="_id",
            cluster=client,
            most_recent=True,
        )
        X, y = data_processing.preprocess_and_split_data(df)

        # train test split
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # remove outliers
        X_corr, y_corr = preprocessing_transformers.remove_outliers(X_train, y_train)

        print(f"Shape of X_train before outlier removal: {X_train.shape}")
        print(f"Shape of X_train after outlier removal: {X_corr.shape}")

        # train model
        regressor = model.create_tuned_pipeline(X_corr, y_corr)

        # evaluate model and save results to database
        model_evaluation = model.evaluate_model(
            regressor, X_corr, y_corr, X_test, y_test
        )
        print(model_evaluation)
        db.model_performance.insert_one(model_evaluation)

        # fit mapie model and save model
        mapie_model = mapie.regression.MapieRegressor(regressor, method="base", cv=5)
        mapie_model.fit(X_corr, y_corr)
        joblib.dump(mapie_model, utils.Configuration.MODEL.joinpath("mapie_model.pkl"))

    finally:
        client.close()
        session.close()


if __name__ == "__main__":
    main()
