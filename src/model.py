from datetime import date
from typing import Dict, Optional, Union

import catboost
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import (
    compose,
    feature_selection,
    impute,
    metrics,
    model_selection,
    pipeline,
    preprocessing,
)
from sklearn.base import BaseEstimator

import preprocessing_transformers
import utils


def create_tuned_pipeline(
    X_train: pd.DataFrame, y_train: pd.Series, random_seed: Optional[int] = None
) -> model_selection.RandomizedSearchCV:
    """
    This function creates a tuned pipeline for a CatBoostRegressor model. It preprocesses the data,
    applies feature selection, trains the model, and performs hyperparameter tuning using randomized search.

    Parameters:
    - X_train (pd.DataFrame): The training data.
    - y_train (pd.Series): The target values for the training data.
    - random_seed (int, optional): The seed for the random number generator. Defaults to None.

    Returns:
    - model_selection.RandomizedSearchCV: The tuned model.
    """
    NUMERICAL_FEATURES = X_train.select_dtypes("number").columns.tolist()
    CATEGORICAL_FEATURES = X_train.select_dtypes("object").columns.tolist()

    optimal_bins = int(np.floor(np.log2(X_train.shape[0])) + 1)

    categorical_column_transformer = pipeline.make_pipeline(
        preprocessing_transformers.CategoricalColumnTransformer(
            categorical_feature="building_condition",
            numerical_feature="construction_year",
            transform_type="mean",
        ),
        impute.SimpleImputer(strategy="median"),
    )

    continuous_discretizer = pipeline.make_pipeline(
        preprocessing_transformers.ContinuousColumnTransformer(
            continuous_feature_to_bin="bathrooms",
            continuous_feature_to_transfer="number_of_frontages",
            transform_type="mean",
            n_bins=optimal_bins,
        ),
        impute.SimpleImputer(strategy="median"),
    )

    polyfeatures = pipeline.make_pipeline(
        impute.SimpleImputer(strategy="median"),
        preprocessing.PolynomialFeatures(interaction_only=False, include_bias=False),
    )
    numeric_transformer = pipeline.make_pipeline(
        impute.SimpleImputer(strategy="median"),
    )

    categorical_transformer = pipeline.make_pipeline(
        impute.SimpleImputer(strategy="most_frequent"),
        preprocessing.OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=999
        ),
    )

    # Create a ColumnTransformer to handle both numerical and categorical features
    transformers = [
        (numeric_transformer, NUMERICAL_FEATURES),
        (categorical_transformer, CATEGORICAL_FEATURES),
        (categorical_column_transformer, ["building_condition", "construction_year"]),
        (continuous_discretizer, ["bathrooms", "number_of_frontages"]),
        (polyfeatures, ["bathrooms"]),
    ]

    preprocessor = compose.make_column_transformer(*transformers).set_output(
        transform="pandas"
    )

    model_pipeline = pipeline.make_pipeline(
        preprocessor,
        feature_selection.VarianceThreshold(),
        catboost.CatBoostRegressor(
            iterations=utils.Configuration.CATBOOST_ITERATIONS,
            eval_fraction=utils.Configuration.CATBOOST_EVAL_FRACTION,
            early_stopping_rounds=utils.Configuration.CATBOOST_EARLY_STOPPING_ROUNDS,
            silent=True,
            use_best_model=True,
            random_seed=utils.Configuration.RANDOM_SEED,
        ),
    )

    param_distributions = {
        "variancethreshold__threshold": stats.uniform(0, 1),
    }

    grid = model_selection.RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_distributions,
        scoring="neg_root_mean_squared_error",
        n_iter=utils.Configuration.RANDCV_ITERATIONS,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    return grid.best_estimator_


def evaluate_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Union[str, float]]:
    """
    This function evaluates a model by calculating the average validation score and the average test score.
    It uses negative root mean squared error as the scoring metric and cross-validation for the validation score.

    Parameters:
    - model (BaseEstimator): The model to be evaluated.
    - X_train (pd.DataFrame): The training data.
    - y_train (pd.Series): The target values for the training data.
    - X_test (pd.DataFrame): The test data.
    - y_test (pd.Series): The target values for the test data.

    Returns:
    - Dict[str, Union[str, float]]: A dictionary containing the day of retrieval, the average validation score, and the average test score.
    """
    AVG_val_score = -np.mean(
        model_selection.cross_val_score(
            estimator=model,
            X=X_train,
            y=y_train,
            scoring="neg_root_mean_squared_error",
            cv=utils.Configuration.CROSSVAL_FOLDS,
        )
    )
    AVG_test_score = metrics.root_mean_squared_error(y_test, model.predict(X_test))

    return {
        "day_of_retrieval": str(date.today()),
        "AVG_val_score": AVG_val_score,
        "AVG_test_score": AVG_test_score,
    }
