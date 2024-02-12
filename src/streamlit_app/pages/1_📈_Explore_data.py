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


@st.cache_data
def retrieve_data_and_exclude_columns(
    db_name, collection_name, query, columns_to_exclude
):
    cluster = MongoClient(creds.Creds.URI)
    db = cluster[db_name]
    collection = db[collection_name]
    df = find_pandas_all(collection, query)
    df = df.drop(columns=columns_to_exclude)
    return df


df = retrieve_data_and_exclude_columns(
    "development", "BE_houses", {"day_of_retrieval": "2024-02-09"}, "_id"
)


def format_number(num):
    if num > 1000000:
        if not num % 1000000:
            return f"{num // 1000000} M"
        return f"{round(num / 1000000, 1)} M"
    return f"{int(num // 1000)} K"


col1, col2, col3 = st.columns([1, 1, 1], gap="small")

with col1:
    BE_provinces = requests.get(
        "https://raw.githubusercontent.com/mathiasleroy/Belgium-Geographic-Data/master/dist/polygons/be-provinces-unk-WGS84.geo.json"
    ).json()
    aggregate = (
        df.assign(list_price=lambda df: pd.to_numeric(df.price))
        .groupby("province")
        .agg(
            list_price_count=("price", "count"),
            list_price_mean=("price", "median"),
        )
        .reset_index()
    )
    fig = px.choropleth(
        aggregate,
        geojson=BE_provinces,
        locations="province",
        color="list_price_mean",
        featureidkey="properties.name",
        projection="mercator",
        color_continuous_scale="Magenta",
        labels={
            "list_price_mean": "Median Price",
            "list_price_count": "Number of Observations",
        },
        hover_data={
            "list_price_mean": ":.3s",
            "province": True,
            "list_price_count": True,
        },
    )

    fig.update_geos(
        showcountries=True, showcoastlines=True, showland=True, fitbounds="locations"
    )

    # Add title and labels
    fig.update_layout(
        title_text="Median House Prices by Province",
        title_font=dict(size=24),
        coloraxis_showscale=False,
        autosize=True,
        width=500,
        height=600,
        margin=dict(l=0, r=0, b=0, t=30),
        geo=dict(showframe=False, showcoastlines=False, projection_type="mercator"),
    )
    st.plotly_chart(fig)

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
    pass


with col3:
    pass
