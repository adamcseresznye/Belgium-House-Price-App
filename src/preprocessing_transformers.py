from typing import Tuple

import catboost
import pandas as pd
from cleanlab.regression.learn import CleanLearning
from sklearn import compose, impute, pipeline, preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

import utils


class CategoricalColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, categorical_feature: str, numerical_feature: str, transform_type: str
    ):
        """
        Initializes a custom Scikit-learn column transformer. This transformer applies a groupby operation on the
        categorical_feature and performs a transformation on the numerical_feature based on the transform_type.

        Parameters:
        - categorical_feature (str): The name of the categorical feature.
        - numerical_feature (str): The name of the numerical feature.
        - transform_type (str): The type of transformation to apply.
        """
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
        continuous_feature_to_bin: str,
        continuous_feature_to_transfer: str,
        transform_type: str,
        n_bins: int,
    ):
        """
        Initializes a custom Scikit-learn column transformer. This transformer first converts the
        continuous_feature_to_bin into a categorical feature followed by a groupby operation. Then it
        performs a transformation on the continuous_feature_to_transfer based on the transform_type.

        Parameters:
        - continuous_feature_to_bin (str): The name of the continuous feature to bin.
        - continuous_feature_to_transfer (str): The name of the continuous feature to transfer.
        - transform_type (str): The type of transformation to apply.
        - n_bins (int): The number of bins to use when binning the continuous feature.
        """
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


def remove_outliers(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    This function removes outliers from the input DataFrame and target Series using the
    CleanLab library (https://github.com/cleanlab/cleanlab).
    It uses a pipeline of transformers to preprocess the data, fits a CatBoostRegressor model,
    identifies label issues, and removes rows corresponding to these issues from the data.

    Parameters:
    - X (pd.DataFrame): The input DataFrame.
    - y (pd.Series): The target Series.

    Returns:
    - Tuple[pd.DataFrame, pd.Series]: A tuple containing the DataFrame and Series with outliers removed.
    """
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
            iterations=utils.Configuration.CATBOOST_ITERATIONS / 10,
            eval_fraction=utils.Configuration.CATBOOST_EVAL_FRACTION,
            early_stopping_rounds=utils.Configuration.CATBOOST_EARLY_STOPPING_ROUNDS,
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
