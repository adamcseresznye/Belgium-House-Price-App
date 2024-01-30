import os
import re
import time
from datetime import date
from io import StringIO

import pandas as pd
import pymongo
from requests_html import HTMLSession


def get_house_urls(url, all_links=None):
    if all_links is None:
        all_links = []

    session = HTMLSession(
        browser_args=[
            "--no-sandbox",
            "--user-agent=Mozilla/5.0 (Windows NT 5.1; rv:7.0.1) Gecko/20100101 Firefox/7.0.1",
        ]
    )
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
        return get_house_urls(next_page, all_links)
    else:
        return all_links


def get_house_data(url):
    try:
        session = HTMLSession()
        r = session.get(url)
        r.html.render(timeout=15)

        individual_ad = pd.concat(pd.read_html(StringIO(r.text))).dropna().set_index(0)

        individual_ad.loc["day_of_retrieval", 1] = str(date.today())
        individual_ad.loc["ad_url", 1] = url
        individual_ad.loc["zip_code", 1] = re.search(r"/(\d{4})/", url).group(1)

        return individual_ad
    except Exception as e:
        print(f"An error occurred: {e}")


def process_item(data):
    # List of columns to keep
    columns_to_keep = [
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

    return (
        data.to_frame()
        .rename(index=lambda x: x.replace(" ", "_").lower())
        .filter(items=columns_to_keep, axis="index")
        .transpose()
        .assign(
            price=lambda x: x["price"]
            .str.extract(r"(\d+,\d+)", expand=False)
            .str.replace(",", "")
            .astype(int)
        )
        .transpose()
    )


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


if __name__ == "__main__":
    """
    # get links to houses
    urls = get_house_urls(
        "https://www.immoweb.be/en/search/house/for-sale?countries=BE&page=332&orderBy=relevance"
    )

    # connect to mongodb
    mongo_uri = os.getenv("MONGO_URI")
    client = pymongo.MongoClient(mongo_uri)
    db = client.test

    for url in urls:
        print("Working on:", url)
        result = get_house_data(url)
        db.BE_houses.insert_one(result)
        time.sleep(1)

    client.close()
    """

    d = {
        "Available as of": "After signing the deed",
        "Available date": "April 1 2024 - 12:00 AM",
        "Construction year": "1900",
        "Building condition": "Good",
        "Street frontage width": "30 m",
        "Number of frontages": "4",
        "Outdoor parking spaces": "10",
        "Surroundings type": "Isolated",
        "Living area": "180  m² square meters",
        "Living room surface": "44  m² square meters",
        "Dining room": "Yes",
        "Kitchen type": "Installed",
        "Kitchen surface": "12  m² square meters",
        "Bedrooms": "4",
        "Bedroom 1 surface": "10  m² square meters",
        "Bedroom 2 surface": "10  m² square meters",
        "Bedroom 3 surface": "13  m² square meters",
        "Bedroom 4 surface": "17  m² square meters",
        "Bathrooms": "1",
        "Toilets": "2",
        "Office surface": "8  m² square meters",
        "Office": "Yes",
        "Professional space surface": "93  m² square meters",
        "Basement surface": "25  m² square meters",
        "Furnished": "No",
        "Surface of the plot": "2435  m²  square meters",
        "Connection to sewer network": "Connected",
        "Garden surface": "2000  m²  square meters",
        "Terrace surface": "30  m²  square meters",
        "Primary energy consumption": "329  kWh/m²  kilowatt hour per square meters",
        "Energy class": "D",
        "Reference number of the EPC report": "20231117000919",
        "CO₂ emission": "81 kg CO₂/m²",
        "Yearly theoretical total energy consumption": "77396 kWh/year",
        "Heating type": "Fuel oil",
        "Planning permission obtained": "Yes",
        "Total ground floor buildable": "300  m²  square meters",
        "Subdivision permit": "Yes",
        "Latest land use designation": "Living area (residential, urban or rural)",
        "Price": "€ 400,000  400000 €",
        "Cadastral income": "€ 468  468 €",
        "Tenement building": "No",
        "Address": "Chaussée de Tirlemont 15/1 4260  - Braives",
        "Website": "http://www.immodemarneffe.be",
        "External reference": "5619624",
        "day_of_retrieval": "2024-01-30",
        "ad_url": "https://www.immoweb.be/en/classified/house/for-sale/braives/4260/10985247",
        "zip_code": "4260",
    }
    print(process_item(pd.Series(d)))
