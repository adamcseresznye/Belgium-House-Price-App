import os
import re
import sys
import time
from datetime import date
from io import StringIO

import numpy as np
import pandas as pd
import pymongo
from requests_html import HTMLSession


def get_house_urls(
    url="https://www.immoweb.be/en/search/house/for-sale?countries=BE&page=1&orderBy=relevance",
    all_links=None,
    session=None,
    N=50,  # Close and reopen the session after every N pages
):
    if all_links is None:
        all_links = []

    r = session.get(url)
    r.html.render(timeout=15)

    elements = r.html.find(".search-results a")
    links = [element.attrs["href"] for element in elements if "href" in element.attrs]
    filtered_links = [
        link
        for link in links
        if link and "www.immoweb.be/en/classified/house/for-sale" in link
    ]

    all_links.extend(filtered_links)

    next_page = r.html.next()
    if next_page:
        print("Next page:", next_page)
        if len(all_links) % N == 0:
            session.close()
            session = HTMLSession(
                browser_args=[
                    "--no-sandbox",
                    "--user-agent=Mozilla/5.0 (Windows NT 5.1; rv:7.0.1) Gecko/20100101 Firefox/7.0.1",
                ]
            )
        return get_house_urls(next_page, all_links, session)
    else:
        return all_links


def get_house_data(session, url):
    try:
        response = session.get(url)
        response.html.render(timeout=15)

        individual_ad = (
            pd.concat(pd.read_html(StringIO(response.text))).dropna().set_index(0)
        )

        individual_ad.loc["day_of_retrieval", 1] = str(date.today())
        individual_ad.loc["ad_url", 1] = url
        individual_ad.loc["zip_code", 1] = re.search(r"/(\d{4})/", url).group(1)

        return individual_ad
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def convert_zip_to_province(value):
    # data from https://www.spotzi.com/en/data-catalog/categories/postal-codes/belgium/
    if value is None:
        return None

    first_two_digits = int(value[:2])

    province_dict = {
        range(10, 13): "Brussels",
        range(13, 15): "Walloon Brabant",
        range(15, 20): "Flemish Brabant",
        range(30, 35): "Flemish Brabant",
        range(20, 30): "Antwerp",
        range(35, 40): "Limburg",
        range(40, 50): "Liege",
        range(50, 60): "Namur",
        range(60, 66): "Hainaut",
        range(70, 80): "Hainaut",
        range(66, 70): "Luxembourg",
        range(80, 90): "West Flanders",
        range(90, 100): "East Flanders",
    }

    for key in province_dict:
        if first_two_digits in key:
            return province_dict[key]

    return None


class DataCleaner:
    def __init__(self, data):
        self.original_data = data
        self.columns_to_keep = [
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

    def reformat_headers(self, data):
        return data.transpose().rename(columns=lambda x: x.replace(" ", "_").lower())

    def add_missing_columns(self, data):
        for column in self.columns_to_keep:
            if column not in data.columns:
                data[column] = "missing"
        return data

    def filter_columns(self, data):
        return data[self.columns_to_keep]

    def reformat_entries(self, data):
        return (
            data.apply(lambda x: x.astype(str))
            .assign(
                price=lambda x: pd.to_numeric(
                    x["price"]
                    .str.extract(r"(\d+,\d+)", expand=False)
                    .str.replace(",", ""),
                    errors="coerce",
                ),
                province=lambda x: x["zip_code"].apply(convert_zip_to_province),
                energy_class=lambda x: x["energy_class"].str.strip(),
                primary_energy_consumption=lambda x: pd.to_numeric(
                    x["primary_energy_consumption"]
                    .str.extract("(\d+)", expand=False)
                    .str.replace(",", ""),
                    errors="coerce",
                ),
                bedrooms=lambda x: pd.to_numeric(
                    x["bedrooms"].str.extract("(\d+)", expand=False), errors="coerce"
                ),
                tenement_building=lambda x: x["tenement_building"].str.strip(),
                living_area=lambda x: pd.to_numeric(
                    x["living_area"]
                    .str.extract("(\d+)", expand=False)
                    .str.replace(",", ""),
                    errors="coerce",
                ),
                surface_of_the_plot=lambda x: pd.to_numeric(
                    x["surface_of_the_plot"]
                    .str.extract("(\d+)", expand=False)
                    .str.replace(",", ""),
                    errors="coerce",
                ),
                bathrooms=lambda x: pd.to_numeric(
                    x["bathrooms"].str.extract("(\d+)", expand=False), errors="coerce"
                ),
                double_glazing=lambda x: x["double_glazing"].str.strip(),
                number_of_frontages=lambda x: pd.to_numeric(
                    x["number_of_frontages"].str.extract("(\d+)", expand=False),
                    errors="coerce",
                ),
                building_condition=lambda x: x["building_condition"].str.strip(),
                toilets=lambda x: pd.to_numeric(
                    x["toilets"].str.extract("(\d+)", expand=False), errors="coerce"
                ),
                heating_type=lambda x: x["heating_type"].str.strip(),
                construction_year=lambda x: pd.to_numeric(
                    x["construction_year"].str.extract("(\d+)", expand=False),
                    errors="coerce",
                ),
            )
            .transpose()
            .squeeze()
            .to_dict()
        )

    def process_item(self):
        return (
            self.original_data.pipe(self.reformat_headers)
            .pipe(self.add_missing_columns)
            .pipe(self.filter_columns)
            .pipe(self.reformat_entries)
        )


if __name__ == "__main__":
    N = 50

    session = HTMLSession(
        browser_args=[
            "--no-sandbox",
            "--user-agent=Mozilla/5.0 (Windows NT 5.1; rv:7.0.1) Gecko/20100101 Firefox/7.0.1",
        ]
    )
    # get links to houses
    urls = get_house_urls(session=session)
    print("Length of urls:", len(urls))

    # connect to mongodb
    mongo_uri = os.getenv("MONGO_URI")
    client = pymongo.MongoClient(mongo_uri)
    db = client.development
    if client:
        print("Connected to MongoDB")
    else:
        sys.exit("Failed to connect to MongoDB")
    try:
        for i, url in enumerate(urls):
            print(f"Working on: {i} out of {len(urls)} : {url}")
            result = get_house_data(session=session, url=url)

            if result is None:
                print(f"Skipping url due to error: {url}")
                continue

            data_cleaner = DataCleaner(result)
            processed_dict = data_cleaner.process_item()
            print(processed_dict)

            db.BE_houses.insert_one(processed_dict)
            # time.sleep(1)

            if (i + 1) % N == 0:
                session.close()
                session = HTMLSession(
                    browser_args=[
                        "--no-sandbox",
                        "--user-agent=Mozilla/5.0 (Windows NT 5.1; rv:7.0.1) Gecko/20100101 Firefox/7.0.1",
                    ]
                )
    finally:
        client.close()
        session.close()
