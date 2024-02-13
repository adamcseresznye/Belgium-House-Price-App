import catboost
import numpy as np
import pandas as pd
from cleanlab.regression.learn import CleanLearning
from scipy import stats
from sklearn import (
    compose,
    feature_selection,
    impute,
    model_selection,
    pipeline,
    preprocessing,
)
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_feature, numerical_feature, transform_type):
        self.categorical_feature = categorical_feature
        self.numerical_feature = numerical_feature
        self.transform_type = transform_type

    def fit(self, X, y=None):
        # Calculate transformation of numerical_feature based on training data
        self.transform_values_ = X.groupby(self.categorical_feature)[
            self.numerical_feature
        ].agg(self.transform_type)
        return self

    def transform(self, X, y=None):
        # Apply transformation to dataset
        return X.assign(
            CategoricalColumnTransformer=lambda df: df[self.categorical_feature].map(
                self.transform_values_
            )
        )[["CategoricalColumnTransformer"]]

    def get_feature_names_out(self):
        pass


class ContinuousColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        continuous_feature_to_bin,
        continuous_feature_to_transfer,
        transform_type,
        n_bins,
    ):
        self.continuous_feature_to_bin = continuous_feature_to_bin
        self.continuous_feature_to_transfer = continuous_feature_to_transfer
        self.transform_type = transform_type
        self.n_bins = n_bins

    def fit(self, X, y=None):
        # Determine bin edges based on training data
        self.bin_edges_ = pd.qcut(
            x=X[self.continuous_feature_to_bin],
            q=self.n_bins,
            retbins=True,
            duplicates="drop",
        )[1]

        # Calculate transformation of continuous_feature_to_transfer based on training data
        self.transform_values_ = (
            X.assign(
                binned_continuous_feature=lambda df: pd.cut(
                    df[self.continuous_feature_to_bin],
                    bins=self.bin_edges_,
                    labels=False,
                )
            )
            .groupby("binned_continuous_feature")[self.continuous_feature_to_transfer]
            .agg(self.transform_type)
        )
        return self

    def transform(self, X, y=None):
        # Apply binning and transformation to dataset
        return X.assign(
            binned_continuous_feature=lambda df: pd.cut(
                df[self.continuous_feature_to_bin], bins=self.bin_edges_, labels=False
            )
        ).assign(
            ContinuousColumnTransformer=lambda df: df["binned_continuous_feature"].map(
                self.transform_values_
            )
        )[
            ["ContinuousColumnTransformer"]
        ]

    def get_feature_names_out(self):
        pass


def remove_outliers(X, y):
    numerical_cols_idx = [
        X.columns.get_loc(column) for column in X.select_dtypes("number")
    ]
    object_cols_idx = [
        X.columns.get_loc(column) for column in X.select_dtypes("object")
    ]

    numeric_transformer = pipeline.make_pipeline(
        impute.SimpleImputer(strategy="median")
    )

    categorical_transformer = pipeline.make_pipeline(
        impute.SimpleImputer(strategy="most_frequent"),
        preprocessing.OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=999
        ),
    )

    transformers = [
        (numeric_transformer, numerical_cols_idx),
        (categorical_transformer, object_cols_idx),
    ]

    preprocessor = compose.make_column_transformer(*transformers)

    model_pipeline = pipeline.make_pipeline(
        preprocessor,
        catboost.CatBoostRegressor(
            iterations=100,
            eval_fraction=0.2,
            early_stopping_rounds=20,
            silent=True,
            use_best_model=True,
        ),
    )

    cl = CleanLearning(model_pipeline)
    cl.fit(X, y)
    label_issues = cl.get_label_issues()

    X_outliers_removed = X.reset_index(drop=True)[label_issues.is_label_issue == False]
    y_outliers_removed = y.reset_index(drop=True)[label_issues.is_label_issue == False]

    return X_outliers_removed, y_outliers_removed
