import sys
from pathlib import Path
from typing import Tuple

import catboost
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import utils
from models import predict_model

sys.path.append(str(Path(__file__).resolve().parent.parent))


st.set_page_config(
    page_title="Make predictions",
    page_icon="🔎",
    initial_sidebar_state="expanded",
    layout="wide",
    menu_items={
        "Get Help": "https://adamcseresznye.github.io/blog/",
        "Report a bug": "https://github.com/adamcseresznye/house_price_prediction",
        "About": "Explore and Predict Belgian House Prices with Immoweb Data and CatBoost!",
    },
)


@st.cache_resource
def fetch_model():
    """
    Load and return a CatBoost regression model.

    Returns:
        catboost.CatBoostRegressor: The loaded CatBoost regression model.
    """
    model = joblib.load(utils.Configuration.MODEL.joinpath("mapie_model.pkl"))

    return model


st.sidebar.subheader("📢 Get in touch 📢")
cols1, cols2, cols3 = st.sidebar.columns(3)
cols1.markdown(
    "[![Foo](https://cdn3.iconfinder.com/data/icons/picons-social/57/11-linkedin-48.png)](https://www.linkedin.com/in/adam-cseresznye)"
)
cols2.markdown(
    "[![Foo](https://cdn1.iconfinder.com/data/icons/picons-social/57/github_rounded-48.png)](https://github.com/adamcseresznye)"
)
cols3.markdown(
    "[![Foo](https://cdn2.iconfinder.com/data/icons/threads-by-instagram/24/x-logo-twitter-new-brand-48.png)](https://twitter.com/csenye22)"
)

try:
    st.header("Generate Home Price Prediction")
    st.subheader("Input Feature Values")

    st.markdown(
        """Please enter the input features below. While you're _not required_ to provide values for all the listed variables,
                 for the most accurate predictions based on what the model has learned, try to be as specific as possible."""
    )

    with st.expander("click to expand"):
        col1, col2, col3 = st.columns(spec=3, gap="large")

        with col1:
            st.markdown("#### Geography")
            province = st.selectbox(
                "In which region is the house located?",
                (
                    [
                        "Liege",
                        "East Flanders",
                        "Brussels",
                        "Antwerp",
                        "Flemish Brabant",
                        "Walloon Brabant",
                        "Hainaut",
                        "Luxembourg",
                        "West Flanders",
                        "Namur",
                        "Limburg",
                    ]
                ),
            )

            zip_code = st.number_input(
                "What is the zip code of the property?",
                step=1.0,
                format="%.0f",
                value=None,
                placeholder="Type a number...",
            )

        with col2:
            st.markdown("#### Construction")
            bedrooms = st.number_input(
                "How many bedrooms does the property have?",
                step=1.0,
                format="%.0f",
                value=None,
                placeholder="Type a number...",
            )
            toilets = st.number_input(
                "How many toilets does the property have?",
                step=1.0,
                format="%.0f",
                value=None,
                placeholder="Type a number...",
            )
            bathrooms = st.number_input(
                "How many bathrooms does the property have?",
                step=1.0,
                format="%.0f",
                value=None,
                placeholder="Type a number...",
            )
            number_of_frontages = st.number_input(
                "What is the count of frontages for this property?",
                step=1.0,
                format="%.0f",
                value=None,
                placeholder="Type a number...",
            )
            surface_of_the_plot = st.number_input(
                "What is the total land area associated with this property in m2?",
                step=1.0,
                format="%.1f",
                value=None,
                placeholder="Type a number...",
            )
            living_area = st.number_input(
                "What is the living area or the space designated for living within the property in m2?",
                step=1.0,
                format="%.1f",
                value=None,
                placeholder="Type a number...",
            )
            tenement_building = st.selectbox(
                "Is it in a tenement building?", ["Yes", "No"]
            )

        with col3:
            st.markdown("#### Energy")
            primary_energy_consumption = st.number_input(
                "What is the primary energy consumption associated with this property?",
                step=1.0,
                format="%.1f",
                value=None,
                placeholder="Type a number...",
            )
            energy_class = st.selectbox(
                "What is the energy rating the building?",
                ["F", "B", "C", "A", "D", "E", "G", "Not specified", "A+", "A++"],
            )
            double_glazing = st.selectbox(
                "Does the property have double glazing?", ["Yes", "No"]
            )
            heating_type = st.selectbox(
                "What is the heating type of the property?",
                [
                    "Gas",
                    "Electric",
                    "Fuel oil",
                    "missing",
                    "Solar",
                    "Pellet",
                    "Wood",
                    "Carbon",
                ],
            )
            construction_year = st.number_input(
                "What is the construction year of the property?",
                step=1.0,
                format="%.0f",
                value=None,
                placeholder="Type a number...",
            )
            building_condition = st.selectbox(
                "What is the condition of the building?",
                [
                    "To be done up",
                    "As new",
                    "Good",
                    "To renovate",
                    "Just renovated",
                    "To restore",
                ],
            )

    data = {
        "province": [province],
        "zip_code": [zip_code],
        "building_condition": [building_condition],
        "bedrooms": [bedrooms],
        "toilets": [toilets],
        "bathrooms": [bathrooms],
        "number_of_frontages": [number_of_frontages],
        "surface_of_the_plot": [surface_of_the_plot],
        "living_area": [living_area],
        "tenement_building": [tenement_building],
        "double_glazing": [double_glazing],
        "heating_type": [heating_type],
        "construction_year": [construction_year],
        "primary_energy_consumption": [primary_energy_consumption],
        "energy_class": [energy_class],
    }
    X_test = pd.DataFrame(data)

    model = fetch_model()

    click = st.button("Predict house price", key="start-button")

    if click:
        y_pred, y_pis = model.predict(X_test, alpha=0.1)
        st.success(
            f"The property’s estimated price is €{10** y_pred[0]:,.0f}, with a 90% probability of ranging from €{10**y_pis.flatten()[0]:,.0f} to €{10**y_pis.flatten()[1]:,.0f}."
        )


except Exception as e:
    st.error(e)
