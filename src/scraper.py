import re
from datetime import date
from io import StringIO
from typing import List, Optional

import pandas as pd
from requests_html import HTMLSession

import utils


def get_house_urls(
    session: HTMLSession,
    N: int,
    url: str = "https://www.immoweb.be/en/search/house/for-sale?countries=BE&page=1&orderBy=relevance",
    all_links: Optional[List[str]] = None,
) -> List[str]:
    """
    This function recursively scrapes a website for URLs of houses for sale,
    and returns a list of these URLs.

    Parameters:
    - session (HTMLSession): The session used to send HTTP requests.
    - N (int): The number of URLs to scrape before resetting the session.
    - url (str, optional): The URL of the webpage to scrape. Defaults to a specific search on immoweb.be.
    - all_links (List[str], optional): A list of URLs that have already been scraped. Defaults to None.

    Returns:
    - List[str]: A list of URLs of houses for sale.
    """
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
            session = HTMLSession(browser_args=utils.Configuration.BROWSER_ARGS)

        return get_house_urls(session, N, next_page, all_links)
    else:
        return all_links


def get_house_data(session: HTMLSession, url: str) -> pd.DataFrame:
    """
    This function scrapes a specific webpage for data about a house for sale,
    and returns a DataFrame of this data.

    Parameters:
    - session (HTMLSession): The session used to send HTTP requests.
    - url (str): The URL of the webpage to scrape.

    Returns:
    - DataFrame: A DataFrame containing data about the house for sale.
                 If an error occurs during the scraping process, None is returned.
    """
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


def convert_zip_to_province(value: str) -> str:
    """
    This function converts a Belgian postal code into the corresponding province
    based on https://www.spotzi.com/en/data-catalog/categories/postal-codes/belgium/.

    Parameters:
    - value (str, optional): The postal code to convert. If None, the function returns None.

    Returns:
    - str: The name of the province corresponding to the postal code.
           If the postal code does not correspond to any province, or if the input is None, the function returns None.
    """
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
    def __init__(self, data: pd.DataFrame):
        """
        Initializes the DataCleaner object with the original data.

        Parameters:
        - data (pd.DataFrame): The original data to be cleaned.
        """
        self.original_data = data
        self.columns_to_keep = utils.Configuration.DATACLEANER_COLUMNS_TO_KEEP

    def reformat_headers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reformats the headers of the DataFrame by replacing spaces with
        underscores and converting to lowercase.

        Parameters:
        - data (pd.DataFrame): The DataFrame whose headers are to be reformatted.

        Returns:
        - pd.DataFrame: The DataFrame with reformatted headers.
        """
        return data.transpose().rename(columns=lambda x: x.replace(" ", "_").lower())

    def add_missing_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds missing columns to the DataFrame.

        Parameters:
        - data (pd.DataFrame): The DataFrame to which missing columns are to be added.

        Returns:
        - pd.DataFrame: The DataFrame with missing columns added.
        """
        for column in self.columns_to_keep:
            if column not in data.columns:
                data[column] = "missing"
        return data

    def filter_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the DataFrame to keep only the necessary columns.

        Parameters:
        - data (pd.DataFrame): The DataFrame to be filtered.

        Returns:
        - pd.DataFrame: The filtered DataFrame.
        """
        return data[self.columns_to_keep]

    def reformat_entries(self, data: pd.DataFrame) -> dict:
        """
        Reformats the entries in the DataFrame and converts the DataFrame to a dictionary.

        Parameters:
        - data (pd.DataFrame): The DataFrame whose entries are to be reformatted.

        Returns:
        - dict: The reformatted DataFrame as a dictionary.
        """
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

    def process_item(self) -> dict:
        """
        Processes the original data by applying a pipeline of functions.

        Returns:
        - dict: The processed data as a dictionary.
        """
        return (
            self.original_data.pipe(self.reformat_headers)
            .pipe(self.add_missing_columns)
            .pipe(self.filter_columns)
            .pipe(self.reformat_entries)
        )
