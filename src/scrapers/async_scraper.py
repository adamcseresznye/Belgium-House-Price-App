import asyncio
import os
import re
import sys
import time
import warnings
from datetime import date
from io import StringIO

import pandas as pd
import pymongo
from pyppeteer.errors import TimeoutError
from requests_html import AsyncHTMLSession, HTMLSession

warnings.filterwarnings("ignore", category=RuntimeWarning)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


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


BROWSER_ARGS = [
    "--no-sandbox",
    "--user-agent=Mozilla/5.0 (Windows NT 5.1; rv:7.0.1) Gecko/20100101 Firefox/7.0.1",
]


def get_house_urls(
    session,
    N,
    url="https://www.immoweb.be/en/search/house/for-sale?countries=BE&page=1&orderBy=relevance",
    all_links=None,
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
            session = HTMLSession(browser_args=BROWSER_ARGS)

        return get_house_urls(session, N, next_page, all_links)
    else:
        return all_links


async def fetch_and_store_data(session, url, db, sem):
    async with sem:
        response = None
        try:
            response = await session.get(url)
            try:
                await response.html.arender(timeout=15)
            except TimeoutError:
                print(f"Timeout error, skipping this page: {url}")
                return

            individual_ad = (
                pd.concat(pd.read_html(StringIO(response.text))).dropna().set_index(0)
            )

            individual_ad.loc["day_of_retrieval", 1] = str(date.today())
            individual_ad.loc["ad_url", 1] = url
            individual_ad.loc["zip_code", 1] = re.search(r"/(\d{4})/", url).group(1)

            # Process the individual ad
            data_cleaner = DataCleaner(individual_ad)
            processed_dict = data_cleaner.process_item()

            # Insert the processed data into the database
            db.BE_houses.insert_one(processed_dict)

            return processed_dict

        except Exception as e:
            print(f"An error occurred: {e}")
        if response is not None:
            await response.close()


async def main(urls, db, session):
    sem = asyncio.Semaphore(10)
    tasks = [
        asyncio.create_task(fetch_and_store_data(session, url, db, sem)) for url in urls
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results


if __name__ == "__main__":
    N = 100

    regular_session = HTMLSession(browser_args=BROWSER_ARGS)
    # get links to houses
    urls = get_house_urls(
        url="https://www.immoweb.be/en/search/house/for-sale?countries=BE&page=332&orderBy=relevance",
        session=regular_session,
        N=N,
    )
    print("Length of urls:", len(urls))
    regular_session.close()

    async_session = AsyncHTMLSession(browser_args=BROWSER_ARGS)

    # Connect to MongoDB
    mongo_uri = "mongodb+srv://csenyechem:kvmVcLFUXsZtVpmt@cluster0.2ivt0kx.mongodb.net/?retryWrites=true&w=majority"
    # mongo_uri = os.getenv("MONGO_URI")
    client = pymongo.MongoClient(mongo_uri)
    db = client.test
    if client:
        print("Connected to MongoDB")
    else:
        sys.exit("Failed to connect to MongoDB")

    try:
        results = asyncio.run(main(urls, db, async_session))
        print(results)
        print("Ads extracted:", len(results))
    finally:
        client.close()
        async_session.close()
