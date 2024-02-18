import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import pymongo
import streamlit as st

import utils
from data_processing import retrieve_data_from_MongoDB

sys.path.append(str(Path(__file__).resolve().parent.parent))


st.set_page_config(
    page_title="House Price Prediction : Make predictions",
    page_icon="🔎",
    initial_sidebar_state="expanded",
    layout="wide",
    menu_items={
        "Get Help": "https://adamcseresznye.github.io/blog/",
        "Report a bug": "https://github.com/adamcseresznye/Belgium-House-Price-App",
        "About": "Explore and Predict Belgian House Prices with Immoweb Data and CatBoost!",
    },
)


@st.cache_resource
def cached_retrieve_data_from_MongoDB(
    db_name, collection_name, query, columns_to_exclude, _client=None, most_recent=True
):
    return retrieve_data_from_MongoDB(
        db_name,
        collection_name,
        query,
        columns_to_exclude,
        client=_client,
        most_recent=most_recent,
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


def display_model_performance(df):
    melted_df = df.rename(
        columns={"AVG_val_score": "Validation RMSE", "AVG_test_score": "Test RMSE"}
    ).melt(id_vars="day_of_retrieval")
    num_unique_dates = int(melted_df.nunique().day_of_retrieval)

    fig = px.line(
        melted_df,
        "day_of_retrieval",
        "value",
        color="variable",
        title="Model Performance over time",
        labels={
            "variable": "Model Metrics",
            "value": "RMSE Score",
            "day_of_retrieval": "Date of Model Training",
        },
        width=450,
        height=300,
        markers=True,
        line_dash="variable",
    )

    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        nticks=num_unique_dates,
    )
    fig.update_layout(
        legend_title_text="Metrics",
        margin=dict(l=0, r=0, b=0, t=30, pad=0),
        xaxis_title="Date of Model Training",
        yaxis_title="RMSE Score",
        legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="right", x=1),
    )

    return fig


def main():
    try:
        st.header("Generate Home Price Prediction")
        st.subheader("Input Feature Values")

        st.markdown(
            """Please enter the input features below. While you're _not required_ to provide values for all the listed variables,
                    for the most accurate predictions based on what the model has learned, try to be as specific as possible."""
        )
        with st.spinner("Please wait while retrieving data from the database..."):
            mongo_uri = os.getenv("MONGO_URI")
            client = pymongo.MongoClient(mongo_uri)
            historical_model_performance = cached_retrieve_data_from_MongoDB(
                db_name="development",
                collection_name="model_performance",
                query=None,
                columns_to_exclude="_id",
                client=client,
                most_recent=False,
            )
        with st.sidebar:
            with st.expander("See historical model performance"):
                st.plotly_chart(display_model_performance(historical_model_performance))

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
                    index=None,
                )

                zip_code = st.number_input(
                    "What is the zip code of the property?",
                    step=1.0,
                    format="%.0f",
                    value=np.nan,
                    placeholder="Type a number...",
                )

            with col2:
                st.markdown("#### Construction")
                bedrooms = st.number_input(
                    "How many bedrooms does the property have?",
                    step=1.0,
                    format="%.0f",
                    value=np.nan,
                    placeholder="Type a number...",
                )
                toilets = st.number_input(
                    "How many toilets does the property have?",
                    step=1.0,
                    format="%.0f",
                    value=np.nan,
                    placeholder="Type a number...",
                )
                bathrooms = st.number_input(
                    "How many bathrooms does the property have?",
                    step=1.0,
                    format="%.0f",
                    value=np.nan,
                    placeholder="Type a number...",
                )
                number_of_frontages = st.number_input(
                    "What is the count of frontages for this property?",
                    step=1.0,
                    format="%.0f",
                    value=np.nan,
                    placeholder="Type a number...",
                )
                surface_of_the_plot = st.number_input(
                    "What is the total land area associated with this property in m2?",
                    step=1.0,
                    format="%.1f",
                    value=np.nan,
                    placeholder="Type a number...",
                )
                living_area = st.number_input(
                    "What is the living area or the space designated for living within the property in m2?",
                    step=1.0,
                    format="%.1f",
                    value=np.nan,
                    placeholder="Type a number...",
                )
                tenement_building = st.selectbox(
                    "Is it in a tenement building?", ["Yes", "No"], index=None
                )

            with col3:
                st.markdown("#### Energy")
                primary_energy_consumption = st.number_input(
                    "What is the primary energy consumption associated with this property in kWh/m²?",
                    step=1.0,
                    format="%.1f",
                    value=np.nan,
                    placeholder="Type a number...",
                )
                energy_class = st.selectbox(
                    "What is the energy rating the building?",
                    ["F", "B", "C", "A", "D", "E", "G", "Not specified", "A+", "A++"],
                    index=None,
                )
                double_glazing = st.selectbox(
                    "Does the property have double glazing?", ["Yes", "No"], index=None
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
                    index=None,
                )
                construction_year = st.number_input(
                    "What is the construction year of the property?",
                    step=1.0,
                    format="%.0f",
                    value=np.nan,
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
                    index=None,
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

        if click and not X_test.isna().all().all():
            y_pred, y_pis = model.predict(X_test, alpha=0.1)
            st.success(
                f"The property’s estimated price is €{10** y_pred[0]:,.0f}, with a 90% probability of ranging from €{10**y_pis.flatten()[0]:,.0f} to €{10**y_pis.flatten()[1]:,.0f}."
            )
        else:
            st.info("Start filling in the required fields.")

    except Exception as e:
        st.error(e)


if __name__ == "__main__":
    main()
