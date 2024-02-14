# %%NBQA-CELL-SEP9be9eb
import itertools

import catboost
import joblib
import mapie
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cleanlab import regression
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
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

import creds
import data_processing
import model
import preprocessing_transformers
import utils

# %%NBQA-CELL-SEP9be9eb
df = data_processing.retrieve_data_from_MongoDB(
    "development", "BE_houses", {"day_of_retrieval": "2024-02-09"}, "_id"
)


# %%NBQA-CELL-SEP9be9eb
X, y = data_processing.preprocess_and_split_data(df)

print(X.shape, y.shape)


# %%NBQA-CELL-SEP9be9eb
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Shape of X_train : {X_train.shape}, X_test : {X_test.shape}")


# %%NBQA-CELL-SEP9be9eb
NUMERICAL_FEATURES = X_train.select_dtypes("number").columns.tolist()
CATEGORICAL_FEATURES = X_train.select_dtypes("object").columns.tolist()

print(NUMERICAL_FEATURES)
print(CATEGORICAL_FEATURES)


# %%NBQA-CELL-SEP9be9eb
print("Unique values in categorical columns:")
for column in X_train[CATEGORICAL_FEATURES]:
    print(f"{column} : {X_train[column].nunique()}")


# %%NBQA-CELL-SEP9be9eb
def create_pipeline(
    numerical_features, categorical_features, additional_transformers=None
):
    numeric_transformer = pipeline.make_pipeline(
        impute.SimpleImputer(strategy="median")
    )

    categorical_transformer = pipeline.make_pipeline(
        impute.SimpleImputer(strategy="most_frequent"),
        preprocessing.OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=999
        ),
    )

    # Create a ColumnTransformer to handle both numerical and categorical features
    transformers = [
        (numeric_transformer, numerical_features),
        (categorical_transformer, categorical_features),
    ]

    if additional_transformers is not None:
        transformers.extend(additional_transformers)

    preprocessor = compose.make_column_transformer(*transformers).set_output(
        transform="pandas"
    )

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

    return model_pipeline


create_pipeline(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)


# %%NBQA-CELL-SEP9be9eb
scores = model_selection.cross_validate(
    estimator=create_pipeline(NUMERICAL_FEATURES, CATEGORICAL_FEATURES),
    X=X_train,
    y=y_train,
    scoring=("r2", "neg_root_mean_squared_error"),
    cv=10,
)


# %%NBQA-CELL-SEP9be9eb
print(
    f'OOF RMSE of basic pipeline : {np.mean(scores["test_neg_root_mean_squared_error"])}'
)
print(f'OOF R2 of basic pipeline : {np.mean(scores["test_r2"])}')


# %%NBQA-CELL-SEP9be9eb
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


# %%NBQA-CELL-SEP9be9eb
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


# %%NBQA-CELL-SEP9be9eb
class EmpiricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Apply transformation to dataset
        return X.assign(
            energy_efficiency=lambda df: df.primary_energy_consumption / df.living_area,
            total_rooms=lambda df: df.bathrooms + df.bedrooms,
            bedroom_to_bathroom=lambda df: df.bedrooms / df.bathrooms,
            area_per_room=lambda df: df.living_area / df.bedrooms,
            plot_to_livings_area=lambda df: df.surface_of_the_plot / df.living_area,
        ).loc[:, "energy_efficiency":]

    def get_feature_names_out(self):
        pass


# %%NBQA-CELL-SEP9be9eb
pd.DataFrame(
    data={
        "condition": [
            "Utilize categorical columns for grouping",
            "Generate bins from the continuous variables",
            "Introduce polynomial features",
            "Empirical observations",
            "Original",
        ],
        "mean_OOFs": [0.083537, 0.083759, 0.084046, 0.084209, 0.08467],
    }
).sort_values(by="mean_OOFs").reset_index(drop=True)


# %%NBQA-CELL-SEP9be9eb
X_corr, y_corr = preprocessing_transformers.remove_outliers(X_train, y_train)

print(f"Shape of X_train before correction: {X_train.shape}")
print(f"Shape of X_train after correction: {X_corr.shape}")


# %%NBQA-CELL-SEP9be9eb
regressor = model.create_tuned_pipeline(X_corr, y_corr)


# %%NBQA-CELL-SEP9be9eb
scores = model_selection.cross_val_score(
    estimator=regressor,
    X=X_corr,
    y=y_corr,
    scoring="neg_root_mean_squared_error",
    # scoring="r2",
    cv=10,
)
np.mean(scores)


# %%NBQA-CELL-SEP9be9eb
mapie_model = mapie.regression.MapieRegressor(regressor, method="base", cv=5)
# Fit the MAPIE model
mapie_model.fit(X_corr, y_corr)
joblib.dump(mapie_model, utils.Configuration.MODEL.joinpath("mapie_model.pkl"))

mapie_model = joblib.load(utils.Configuration.MODEL.joinpath("mapie_model.pkl"))


# %%NBQA-CELL-SEP9be9eb
# Make predictions with prediction intervals on the transformed validation data
y_pred, y_pis = mapie_model.predict(X_corr, alpha=0.1)


# %%NBQA-CELL-SEP9be9eb
# Create a DataFrame with y_valid and prediction intervals
conformal_df = 10 ** pd.DataFrame(
    {
        "y_test": y_test,
        "lower": y_pis[:, 0].flatten(),
        "upper": y_pis[:, 1].flatten(),
        "y_pred": y_pred,
    }
)

# Sort the DataFrame by y_valid
df_sorted = conformal_df.sort_values(by="y_test")

# Plot data

plt.scatter(
    range(df_sorted.shape[0]),
    df_sorted["y_pred"],
    color="red",
    label="predicted",
    alpha=0.2,
)
plt.scatter(
    range(df_sorted.shape[0]),
    df_sorted["y_test"],
    color="green",
    label="ground truth",
    alpha=0.1,
)
plt.fill_between(
    range(df_sorted.shape[0]),
    df_sorted["lower"],
    df_sorted["upper"],
    alpha=0.2,
    color="gray",
    label="Prediction Intervals",
)

plt.legend()


# %%NBQA-CELL-SEP9be9eb
X
