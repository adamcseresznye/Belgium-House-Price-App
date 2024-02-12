import catboost
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import (
    compose,
    feature_selection,
    impute,
    model_selection,
    pipeline,
    preprocessing,
)

import feature_pipeline


def create_tuned_pipeline(X_train, y_train, random_seed=None):
    NUMERICAL_FEATURES = X_train.select_dtypes("number").columns.tolist()
    CATEGORICAL_FEATURES = X_train.select_dtypes("object").columns.tolist()

    optimal_bins = int(np.floor(np.log2(X_train.shape[0])) + 1)

    categorical_column_transformer = pipeline.make_pipeline(
        feature_pipeline.CategoricalColumnTransformer(
            categorical_feature="building_condition",
            numerical_feature="construction_year",
            transform_type="mean",
        ),
        impute.SimpleImputer(strategy="median"),
    )

    continuous_discretizer = pipeline.make_pipeline(
        feature_pipeline.ContinuousColumnTransformer(
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
            iterations=1000,
            eval_fraction=0.2,
            early_stopping_rounds=20,
            silent=True,
            use_best_model=True,
            random_seed=random_seed,
        ),
    )

    param_distributions = {
        "variancethreshold__threshold": stats.uniform(0, 1),
    }

    grid = model_selection.RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_distributions,
        scoring="neg_root_mean_squared_error",
        n_iter=10,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    return grid.best_estimator_
