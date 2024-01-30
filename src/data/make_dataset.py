import os
import time
from datetime import date
from io import StringIO

import pandas as pd
import pymongo
from requests_html import Element, HTMLSession
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager


def get_house_urls(url):
    options = Options()
    options.headless = True

    service = Service(GeckoDriverManager().install())
    driver = webdriver.Firefox(service=service, options=options)

    try:
        next_page = url

        all_links = []

        while next_page:
            driver.get(next_page)
            print("Scraping:", next_page)

            elements = driver.find_elements(By.CSS_SELECTOR, ".search-results a")
            links = [element.get_attribute("href") for element in elements]
            filtered_links = [
                link
                for link in links
                if link and "www.immoweb.be/en/classified/house/for-sale" in link
            ]

            all_links.extend(filtered_links)

            # Find the link to the next page
            next_page_elements = driver.find_elements(
                By.CSS_SELECTOR, "a.pagination__link--next"
            )
            if next_page_elements:
                next_page = next_page_elements[0].get_attribute("href")
            else:
                break
    finally:
        driver.quit()

    return all_links


def get_house_data(url):
    session = HTMLSession()
    r = session.get(url)
    r.html.render(sleep=5)

    individual_ad = pd.concat(pd.read_html(StringIO(r.text))).dropna().set_index(0)

    # Extract list_price and province
    list_price = r.html.find("p.classified__price span.sr-only", first=True)
    if list_price:
        individual_ad.loc["list_price", 1] = list_price.text.strip()

    individual_ad.loc["day_of_retrieval", 1] = str(date.today())
    individual_ad.loc["ad_url", 1] = url
    return individual_ad.squeeze().to_dict()


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

    client.close()
