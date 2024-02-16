import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import utils

sys.path.append(str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="House Price Prediction : Intro",
    page_icon="📄",
    initial_sidebar_state="expanded",
    layout="wide",
    menu_items={
        "Get Help": "https://adamcseresznye.github.io/blog/",
        "Report a bug": "https://github.com/adamcseresznye/house_price_prediction",
        "About": "Explore and Predict Belgian House Prices with Immoweb Data and CatBoost!",
    },
)


def main():
    st.write("# House Price Prediction")

    st.markdown(
        """
        [![Star](https://img.shields.io/github/stars/adamcseresznye/Belgian-House-Price-Predictor)](https://gitHub.com/adamcseresznye/Belgian-House-Price-Predictor)
        [![Follow](https://img.shields.io/twitter/follow/csenye22?style=social)](https://www.twitter.com/csenye22)

        This app is designed to predict house prices in Belgium using data gathered from [immoweb.be](https://www.immoweb.be/en), a prominent real
        estate platform in the country. Leveraging a CatBoost model with MAPIE, we aim to offer accurate and current price predictions with prediction intervals based on conformal prediction.
        Explore the housing market and make informed choices with our prediction tool.
    """
    )
    st.subheader("Introduction")

    file_path = Path(__file__).parent
    image_path = file_path.joinpath("diagram.png")
    image = Image.open(image_path)

    st.subheader("Describing the Workflow")
    st.info(
        """
        This is version 2 of our app, completely redesigned from the ground up.

        **What has changed:**
        - Data that has been scraped and sanitized is now directly uploaded to our MongoDB database.
        - The unnecessary features have been eliminated.
        - Automatic feature selection is included with hyperparameter optimization using GridSearchCV.
        - Predictions are made with confidence intervals at the 90% level, utilizing conformal prediction.

        For an in-depth exploration of the pipeline's creation of the first version of our app, encompassing
        [data preparation](https://adamcseresznye.github.io/blog/projects/Predicting%20Belgian%20Real%20Estate%20Prices_%20Part%201_Feature%20Selection%20for%20Web%20Scraping/NB_1_ACs_Select_features_for_scraping.html),
        [cleaning](https://adamcseresznye.github.io/blog/projects/Predicting%20Belgian%20Real%20Estate%20Prices_%20Part%202_Basic_clean_up_after_scraping/NB_2_ACs_Basic_clean_up_after_scraping.html),
        [feature generation](https://adamcseresznye.github.io/blog/projects/Predicting%20Belgian%20Real%20Estate%20Prices_%20Part%206_Feature_engineering/NB_6_ACs_Feature_engineering.html),
        [feature importance assessment](https://adamcseresznye.github.io/blog/projects/Predicting%20Belgian%20Real%20Estate%20Prices_%20Part%205_Initial_feature_selection/NB_5_ACs_Initial_feature_selection.html),
        [model training](https://adamcseresznye.github.io/blog/projects/Predicting%20Belgian%20Real%20Estate%20Prices_%20Part%204_Building_a_baseline_model/NB_4_ACs_Building_a_baseline_model.html),
        [fine-tuning](https://adamcseresznye.github.io/blog/projects/Predicting%20Belgian%20Real%20Estate%20Prices_%20Part%207_Fine_tuning/NB_7_ACs_Fine_tuning.html)
        and more, feel free to explore my
        [seven-part series of articles](https://adamcseresznye.github.io/blog/projects.html).
            """
    )
    st.image(
        image,
        caption="Data Acquisition, Processing, Model Training, and Performance Testing Workflow.",
        width=800,
    )
    st.markdown(
        """From the diagram, you can see that our data processing pipeline adheres to the traditional Extract, Transform, and Load (ETL) process.
                Initially, we extract data from the source using the `request_html` library. Following this, we execute multiple steps to refine the raw data,
                encompassing datatype conversions from strings to numerical values and converting low cardinal numerical data to boolean. Furthermore, we leverage
                the Google Maps API to enhance location precision and obtain coordinates based on the provided location information in the advertisements.
                """
    )
    st.markdown(
        """Once our data is prepared, we perform the essential step of splitting it into train and test sets. This step ensures an unbiased model
                performance evaluation later on. It's worth noting that during the project's experimental phases, we
                [evaluated](https://adamcseresznye.github.io/blog/projects/Predicting%20Belgian%20Real%20Estate%20Prices_%20Part%204_Building_a_baseline_model/NB_4_ACs_Building_a_baseline_model.html)
                various ML algorithms, including decision trees, `XGBoost`, and more. After rigorous experimentation, we selected `Catboost` due to
                its exceptional overall performance."""
    )
    st.markdown(
        """Upon loading our pre-defined and optimized hyperparameters, we are fully equipped to train and subsequently assess the model's performance using the test set.
                The results of the model's performance are saved for reference.
                This entire pipeline is initiated on a monthly basis through GitHub actions, ensuring that both our model and dataset remain up-to-date."""
    )
    st.subheader("How to Use This App")
    st.markdown(
        """Visit the _"Explore data"_ page to gain insights into the factors influencing house prices in Belgium. Once you've grasped these principles,
                feel free to make predictions using the features available on the _"Make predictions"_ page and assess your accuracy based on our model.
                Have fun!🎈🎉😊🐍💻🎈
        """
    )
    st.subheader("Planned Enhancements")
    st.info(
        """
        - :white_check_mark: Collaborate with the [Kaggle](https://www.kaggle.com/datasets/unworried1686/belgian-property-prices-2023/data) community. Share initial data and gather insights on potential model improvements and data preprocessing techniques for better predictions.
        - :white_check_mark: Incorporate confidence intervals into predictions.
        - :white_check_mark: Implement data upload to a database for improved data management.
        - :construction: Explore the inclusion of advertisement time to account for seasonality in the model.
        - :construction: Make the scraping process faster using asynchronous operations.
    """
    )

    st.caption(
        """Disclaimer: The developer is not liable for the information provided by this app.
            It is intended for educational purposes only. Any use of the information for decision-making or financial
            purposes is at your own discretion and risk."""
    )


if __name__ == "__main__":
    main()
