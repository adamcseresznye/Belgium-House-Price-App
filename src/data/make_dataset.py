import os
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

        # Extract list_price and province
        list_price = r.html.find("p.classified__price span.sr-only", first=True)
        if list_price:
            individual_ad.loc["list_price", 1] = list_price.text.strip()

        individual_ad.loc["day_of_retrieval", 1] = str(date.today())
        individual_ad.loc["ad_url", 1] = url
        return individual_ad.squeeze().to_dict()
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
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
