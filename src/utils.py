from pathlib import Path


class Configuration:
    VER = 1
    RAW_DATA_PATH = Path(__file__).parents[1].joinpath("data/raw")
    INTERIM_DATA_PATH = Path(__file__).parents[1].joinpath("data/interim")
    MODEL = Path(__file__).parents[1].joinpath("models")
    BROWSER_ARGS = [
        "--no-sandbox",
        "--user-agent=Mozilla/5.0 (Windows NT 5.1; rv:7.0.1) Gecko/20100101 Firefox/7.0.1",
    ]
    MAX_NUMBER_OF_PAGES = 100
    DATACLEANER_COLUMNS_TO_KEEP = [
        "ad_url",
        "price",
        "day_of_retrieval",
        "zip_code",
        "energy_class",
        "primary_energy_consumption",
        "bedrooms",
        "tenement_building",
        "living_area",
        "surface_of_the_plot",
        "bathrooms",
        "double_glazing",
        "number_of_frontages",
        "building_condition",
        "toilets",
        "heating_type",
        "construction_year",
    ]
    DB_NAME = "development"
    COLLECTION_NAME_DATA = "BE_houses"
    COLLECTION_NAME_RMSE = "model_performance"
    TARGET_COLUMN = "price"
    RANDOM_SEED = 3407
    CATBOOST_ITERATIONS = 1000
    CATBOOST_EVAL_FRACTION = 0.2
    CATBOOST_EARLY_STOPPING_ROUNDS = 20
    RANDCV_ITERATIONS = 10
    CROSSVAL_FOLDS = 10
