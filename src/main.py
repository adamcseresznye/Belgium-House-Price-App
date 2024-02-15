import os

import joblib
import mapie
import numpy as np
import pymongo
from sklearn import model_selection

import data_processing
import model
import preprocessing_transformers
import scrapers
import utils

if __name__ == "__main__":
    # connect to mongodb
    mongo_uri = os.getenv("MONGO_URI")
    client = pymongo.MongoClient(mongo_uri)
    db = client.development

    # get links to houses
    # here comes to the scraping part

    # get data and preprocess
    df = data_processing.retrieve_data_from_MongoDB(
        "development", "BE_houses", None, "_id"
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
    model_evaluation = model.evaluate_model(regressor, X_corr, y_corr, X_test, y_test)
    db.model_performance.insert_one(model_evaluation)

    # fit mapie model and save model
    mapie_model = mapie.regression.MapieRegressor(regressor, method="base", cv=5)
    mapie_model.fit(X_corr, y_corr)
    joblib.dump(mapie_model, utils.Configuration.MODEL.joinpath("mapie_model.pkl"))
