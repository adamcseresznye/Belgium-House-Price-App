import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import statsmodels.api as sm
import streamlit as st
import streamlit.components.v1 as components
from pandas.api.types import is_numeric_dtype
from pymongo import MongoClient
from pymongoarrow.api import find_pandas_all
from sklearn import compose, impute, neighbors, pipeline, preprocessing

import creds
import data

sys.path.append(str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="Explore the data",
    page_icon="📈",
    initial_sidebar_state="expanded",
    layout="wide",
    menu_items={
        "Get Help": "https://adamcseresznye.github.io/blog/",
        "Report a bug": "https://github.com/adamcseresznye/house_price_prediction",
        "About": "Explore and Predict Belgian House Prices with Immoweb Data and CatBoost!",
    },
)

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

HEIGHT = 400


@st.cache_data
def retrieve_data_and_exclude_columns(
    db_name, collection_name, query, columns_to_exclude
):
    cluster = MongoClient(creds.Creds.URI)
    db = cluster[db_name]
    collection = db[collection_name]
    df = find_pandas_all(collection, query)
    df = df.drop(columns=columns_to_exclude)
    return df.query("price > 100000")


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
        color_continuous_scale="Magenta",
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
        title_font=dict(size=24),
        # coloraxis_showscale=False,
        # width=600,
        height=HEIGHT,
        margin=dict(l=0, r=0, b=0, t=40),
        geo=dict(showframe=False, showcoastlines=False, projection_type="mercator"),
    )
    return fig


def get_stats_plot(data, feature):
    if data[feature].nunique() > 20:
        plot = px.scatter(
            data,
            x=feature,
            y="price",
            trendline="lowess",
            template="plotly_dark",
            title=f"Analyzing the Effect of {feature.replace('_', ' ').title()} on House Prices",
            trendline_color_override="#c91e01",
            opacity=0.5,
            height=HEIGHT,
            labels={
                feature: feature.replace("_", " ").title(),
                "price": "Price in Log10-Scale (EUR)",
            },
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
            title=f"Analyzing the Effect of {feature.replace('_', ' ').title()} on House Prices",
            labels={
                "price": "Price in Log10-Scale (EUR)",
                feature: f"{feature.replace('_', ' ').title()}",
            },
        )
        plot.update_xaxes(categoryorder="array", categoryarray=sorted_index)

    return plot


df = retrieve_data_and_exclude_columns(
    "development", "BE_houses", {"day_of_retrieval": "2024-02-09"}, "_id"
)

BE_provinces = get_BE_provice_map()

selected_feature = st.sidebar.selectbox(
    "Feature to investigate",
    df.select_dtypes("number")
    .drop(columns=["price"])
    .columns.str.replace("_", " ")
    .str.capitalize(),
)

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

    st.subheader("What defines the typical property in Belgium?")
    median_values = df[
        ["price", "primary_energy_consumption", "living_area", "construction_year"]
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
