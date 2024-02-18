import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import pymongo
import requests
import streamlit as st
from pandas.api.types import is_numeric_dtype
from pymongo import MongoClient
from pymongoarrow.api import find_pandas_all

from data_processing import preprocess_and_split_data

sys.path.append(str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="House Price Prediction : Explore the data",
    page_icon="📈",
    initial_sidebar_state="expanded",
    layout="wide",
    menu_items={
        "Get Help": "https://adamcseresznye.github.io/blog/",
        "Report a bug": "https://github.com/adamcseresznye/Belgium-House-Price-App",
        "About": "Explore and Predict Belgian House Prices with Immoweb Data and CatBoost!",
    },
)

HEIGHT = 400
WIDTH = 500
TITLE_FONT_SIZE = 20


@st.cache_resource
def retrieve_data_from_MongoDB(
    db_name: str,
    collection_name: str,
    query: Dict[str, Any],
    columns_to_exclude: Optional[List[str]],
    _client: Optional[MongoClient] = None,
    most_recent: bool = True,
) -> pd.DataFrame:
    """
    This function retrieves data from a MongoDB collection and returns it as a DataFrame.

    Parameters:
    - db_name (str): The name of the MongoDB database.
    - collection_name (str): The name of the collection in the database.
    - query (Dict[str, Any]): The query to use when retrieving data from the collection.
    - columns_to_exclude (List[str], optional): A list of column names to exclude from the DataFrame. If None, no columns are excluded.
    - client (MongoClient, optional): The MongoDB client to use. If None, a new client is created.
    - most_recent (bool, optional): Whether to only include the most recent data in the DataFrame. Defaults to True.

    Returns:
    - pd.DataFrame: The retrieved data as a DataFrame.
    """
    db = _client[db_name]
    collection = db[collection_name]
    df = find_pandas_all(collection, query)
    if columns_to_exclude:
        df = df.drop(columns=columns_to_exclude)
    if most_recent:
        most_recent_values = (
            df.day_of_retrieval.value_counts().sort_values(ascending=True).index[0]
        )
        df = df.query("day_of_retrieval == @most_recent_values")
        return df
    else:
        return df


@st.cache_data
def cached_preprocess_and_split_data(df):
    return preprocess_and_split_data(df)


@st.cache_data
def get_BE_provice_map():
    return requests.get(
        "https://raw.githubusercontent.com/mathiasleroy/Belgium-Geographic-Data/master/dist/polygons/be-provinces-unk-WGS84.geo.json"
    ).json()


def get_choropleth(data, feature, BE_map):
    if feature == "number_of_ads":
        aggregate = (
            data.groupby("province")
            .size()
            .reset_index()
            .rename(columns={0: "number_of_ads"})
        )
        title_text = f"{feature.replace('_', ' ').title()} by Province"
    elif feature == "price":
        aggregate = (
            data.assign(list_price=lambda df: pd.to_numeric(df.price))
            .groupby("province")
            .price.mean()
            .reset_index()
        )
        title_text = f"Average {feature.replace('_', ' ').title()} by Province"
    elif is_numeric_dtype(data[feature]):
        aggregate = data.groupby("province")[feature].mean().reset_index()
        title_text = f"Average {feature.replace('_', ' ').title()} by Province"

    fig = px.choropleth(
        aggregate,
        geojson=BE_map,
        locations="province",
        color=feature,
        featureidkey="properties.name",
        projection="mercator",
        color_continuous_scale="Peach",
    )

    fig.update_geos(
        showcountries=True,
        showcoastlines=True,
        showland=True,
        fitbounds="locations",
    )

    fig.update_layout(
        title_text=title_text,
        coloraxis_colorbar_title_text=f"{feature.replace('_', ' ').title()}",
        autosize=False,
        title_font=dict(size=TITLE_FONT_SIZE),
        width=WIDTH,
        height=HEIGHT,
        margin=dict(l=0, r=0, b=0, t=40),
        geo=dict(showframe=False, showcoastlines=False, projection_type="mercator"),
    )
    return fig


def get_stats_plot(data, feature):
    title = f"Effect of {feature.replace('_', ' ').title()} on House Prices"
    labels = {
        feature: feature.replace("_", " ").title(),
        "price": "Price in Log10-Scale (EUR)",
    }

    if data[feature].nunique() > 20:
        plot = px.scatter(
            data,
            x=feature,
            y="price",
            trendline="lowess",
            template="plotly_dark",
            title=title,
            trendline_color_override="#c91e01",
            opacity=0.5,
            height=HEIGHT,
            width=WIDTH,
            labels=labels,
            log_y=True,
        )
    else:
        sorted_index = data.groupby(feature).price.median().sort_values().index.tolist()
        plot = px.box(
            data,
            x=feature,
            y="price",
            template="plotly_dark",
            log_y=True,
            category_orders=sorted_index,
            height=HEIGHT,
            width=WIDTH,
            title=title,
            labels=labels,
        )
        plot.update_xaxes(categoryorder="array", categoryarray=sorted_index)

    plot.update_layout(
        autosize=False,
        title_font=dict(size=TITLE_FONT_SIZE),
        width=WIDTH,
        height=HEIGHT,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return plot


def main():
    with st.spinner("Please wait while retrieving data from the database..."):
        mongo_uri = os.getenv("MONGO_URI")
        client = pymongo.MongoClient(mongo_uri)

        df = retrieve_data_from_MongoDB(
            db_name="development",
            collection_name="BE_houses",
            query=None,
            columns_to_exclude="_id",
            _client=client,
            most_recent=True,
        )
        day_of_retrieval = df.day_of_retrieval.unique()[0]
        X, y = cached_preprocess_and_split_data(df)
        df = pd.concat([X, 10**y], axis=1)

    BE_provinces = get_BE_provice_map()

    selected_feature = st.sidebar.selectbox(
        "Feature to investigate",
        df.select_dtypes("number")
        .drop(columns=["price", "zip_code"])
        .columns.str.replace("_", " ")
        .str.capitalize(),
        index=None,
        placeholder="Select a feature...",
    )

    if selected_feature is None:
        st.info("Select a feature from the sidebar", icon="ℹ️")

    if selected_feature:
        converted_selected_feature = selected_feature.replace(" ", "_").lower()

        def format_number(num):
            if num > 1000000:
                if not num % 1000000:
                    return f"{num // 1000000} M"
                return f"{round(num / 1000000, 1)} M"
            return f"{int(num // 1000)} K"

        col1, col2, col3 = st.columns([1, 1, 1], gap="small")

        with col1:
            median_price_aggregate_fig = get_choropleth(df, "price", BE_provinces)
            st.plotly_chart(median_price_aggregate_fig)

            st.markdown("#### What defines the typical property in Belgium?")
            median_values = df[
                [
                    "price",
                    "primary_energy_consumption",
                    "living_area",
                    "construction_year",
                ]
            ].median()

            subcol1, subcol2 = st.columns(2, gap="small")
            with subcol1:
                st.metric("Price (€)", format_number(median_values["price"]))
                st.metric(
                    "Energy Consumption (kWh/m²)",
                    int(median_values["primary_energy_consumption"]),
                )

            with subcol2:
                st.metric("Living Area (m²)", int(median_values["living_area"]))
                st.metric("Construction Year", int(median_values["construction_year"]))

        with col2:
            selectable_feature_aggregate_fig = get_choropleth(
                df, converted_selected_feature, BE_provinces
            )
            st.plotly_chart(selectable_feature_aggregate_fig)

            stats_plot = get_stats_plot(df, converted_selected_feature)
            st.plotly_chart(stats_plot)

        with col3:
            number_of_ads_fig = get_choropleth(df, "number_of_ads", BE_provinces)
            st.plotly_chart(number_of_ads_fig)
            st.markdown("#### Fun facts about the dataset")

            oldest_house = df.sort_values(by="construction_year").head(1)
            most_expensive_house = df.sort_values(by="price", ascending=False).head(1)
            energy_ratings = df.energy_class.value_counts(normalize=True)
            energy_ratings_B_above = (
                df.query("energy_class.str.contains('A|B')").province.value_counts()
                / df.province.value_counts()
            ).sort_values(ascending=False)

            st.markdown(
                f"- The dataset, retrieved on **{day_of_retrieval}** contains **{df.shape[0]}** ads."
            )
            st.markdown(
                f" - The oldest house, built in **{int(oldest_house.construction_year.values[0])}**, is in **{oldest_house.zip_code.values[0]}, {oldest_house.province.values[0]}**."
            )
            st.markdown(
                f"- The most expensive house, costing **€{int(most_expensive_house.price.values[0])}**, is in **{most_expensive_house.zip_code.values[0]}, {most_expensive_house.province.values[0]}**."
            )
            st.markdown(
                f"- The most common energy rating is **{energy_ratings.head(1).index[0]}**, accounting for **{energy_ratings.head(1).values[0]:.1%}** of the ads."
            )
            st.markdown(
                f"- **{energy_ratings.filter(regex='A|B', axis = 0).sum():.1%}** of the houses have energy efficiency ratings of **B and above**."
            )
            st.markdown(
                f"- Based on the ratio of houses per province with energy score of B or above, the most energy efficient  province is **{energy_ratings_B_above.head(1).index[0]} ({energy_ratings_B_above.head(1).values[0]:.1%})** while **{energy_ratings_B_above.dropna().tail(1).index[0]} ({energy_ratings_B_above.dropna().tail(1).values[0]:.1%})** is the least."
            )


if __name__ == "__main__":
    main()
