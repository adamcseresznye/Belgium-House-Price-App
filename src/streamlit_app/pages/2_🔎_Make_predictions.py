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
    st.image(
        "https://cf.bstatic.com/xdata/images/hotel/max1024x768/408003083.jpg?k=c49b5c4a2346b3ab002b9d1b22dbfb596cee523b53abef2550d0c92d0faf2d8b&o=&hp=1",
        caption="Photo by Stephen Phillips - Hostreviews.co.uk on UnSplash",
        use_column_width="always",
    )
    st.subheader("Input Feature Values")
    st.markdown(
        """Please enter the input features below. While you're _not required_ to provide values for all the listed variables,
                 for the most accurate predictions based on what the model has learned, try to be as specific as possible."""
    )

    with st.expander("click to expand"):
        col1, col2, col3 = st.columns(spec=3, gap="large")

        with col1:
            st.markdown("#### Geography")
            state = st.selectbox(
                "In which region is the house located?",
                ([0, 1, 2]),
            )

            zip_code = st.number_input(
                "What is the zip code of the property?", step=1.0, format="%.0f"
            )

        with col2:
            st.markdown("#### Construction")
            building_condition = st.selectbox(
                "What is the condition of the building?",
                ([0, 1, 2]),
            )
            bedrooms = st.number_input(
                "How many bedrooms does the property have?", step=1.0, format="%.0f"
            )
            bathrooms = st.number_input(
                "How many bathrooms does the property have?", step=1.0, format="%.0f"
            )
            number_of_frontages = st.number_input(
                "What is the count of frontages for this property?",
                step=1.0,
                format="%.0f",
            )
            surface_of_the_plot = st.number_input(
                "What is the total land area associated with this property in m2?",
                step=1.0,
                format="%.1f",
            )
            living_area = st.number_input(
                "What is the living area or the space designated for living within the property in m2?",
                step=1.0,
                format="%.1f",
            )

        with col3:
            st.markdown("#### Energy, Taxes")
            yearly_theoretical_total_energy_consumption = st.number_input(
                "What is the estimated annual total energy consumption for this property?",
                step=1.0,
                format="%.1f",
            )
            primary_energy_consumption = st.number_input(
                "What is the primary energy consumption associated with this property?",
                step=1.0,
                format="%.1f",
            )
            cadastral_income = st.number_input(
                "What is the cadastral income or property tax assessment value for this property?",
                step=1.0,
                format="%.1f",
            )

    data = {
        "bedrooms": [bedrooms],
        "state": [state],
        "number_of_frontages": [number_of_frontages],
        "primary_energy_consumption": [primary_energy_consumption],
        "bathrooms": [bathrooms],
        "yearly_theoretical_total_energy_consumption": [
            yearly_theoretical_total_energy_consumption
        ],
        "surface_of_the_plot": [surface_of_the_plot],
        "building_condition": [building_condition],
        "city": [city],
        "lat": [lat],
        "cadastral_income": [cadastral_income],
        "living_area": [living_area],
    }
    X_test = pd.DataFrame(data)

    model = fetch_model()

    click = st.button("Predict house price", key="start-button")

    if click:
        prediction = predict_model.predict_catboost(model=model, X=X_test)
        st.success(
            f"The predicted price for the house is {10** prediction[0]:,.0f} EUR."
        )


except Exception as e:
    st.error(e)
