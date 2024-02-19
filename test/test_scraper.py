import re
from datetime import date
from io import StringIO
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from requests_html import HTMLSession

from scraper import DataCleaner, convert_zip_to_province

################ test get_house_urls ###############
####################################################


class MockElement:
    def __init__(self, href):
        self.attrs = {"href": href}


def test_get_house_urls():
    with patch.object(HTMLSession, "get", return_value=Mock()) as mock_get:
        mock_get.return_value.html.render = Mock()
        mock_get.return_value.html.find = Mock(
            return_value=[
                MockElement("www.immoweb.be/en/classified/house/for-sale/test_url")
            ]
        )
        mock_get.return_value.html.next = Mock(return_value=None)

        from scraper import get_house_urls

        session = HTMLSession()
        N = 10
        url = "https://www.immoweb.be/en/search/house/for-sale?countries=BE&page=1&orderBy=relevance"
        all_links = get_house_urls(session, N, url)

        assert len(all_links) == 1
        assert all_links[0] == "www.immoweb.be/en/classified/house/for-sale/test_url"
        mock_get.assert_called_once_with(url)
        mock_get.return_value.html.render.assert_called_once_with(timeout=15)
        mock_get.return_value.html.find.assert_called_once_with(".search-results a")
        mock_get.return_value.html.next.assert_called_once()


@pytest.mark.parametrize("exception", [TimeoutError, Exception])
def test_get_house_urls_exception(exception):
    with patch.object(HTMLSession, "get", side_effect=exception()):
        from scraper import get_house_urls

        session = HTMLSession()
        N = 10
        url = "https://www.immoweb.be/en/search/house/for-sale?countries=BE&page=1&orderBy=relevance"
        with pytest.raises(exception):
            get_house_urls(session, N, url)


################ test get_house_data ###############
####################################################


def test_get_house_data():
    with patch.object(HTMLSession, "get", return_value=Mock()) as mock_get:
        mock_get.return_value.html.render = Mock()
        mock_get.return_value.text = "<table><tr><td>1</td></tr></table>"

        from scraper import get_house_data

        session = HTMLSession()
        url = "https://www.immoweb.be/en/classified/house/for-sale/test_url/1234/another_part"
        df = get_house_data(session, url)

        assert isinstance(df, pd.DataFrame)
        assert df.loc["day_of_retrieval", 1] == str(date.today())
        assert df.loc["ad_url", 1] == url
        assert df.loc["zip_code", 1] == re.search(r"/(\d{4})/", url).group(1)
        mock_get.assert_called_once_with(url)
        mock_get.return_value.html.render.assert_called_once_with(timeout=15)


def test_get_house_data_exception():
    with patch.object(HTMLSession, "get", side_effect=Exception()):
        from scraper import get_house_data

        session = HTMLSession()
        url = "https://www.immoweb.be/en/classified/house/for-sale/test_url/1234"
        df = get_house_data(session, url)
        assert df is None


################ test convert_zip_to_province ###############
####################################################


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        ("1000", "Brussels"),
        ("1300", "Walloon Brabant"),
        ("1500", "Flemish Brabant"),
        ("2000", "Antwerp"),
        ("3500", "Limburg"),
        ("4000", "Liege"),
        ("5000", "Namur"),
        ("6000", "Hainaut"),
        ("6600", "Luxembourg"),
        ("8000", "West Flanders"),
        ("9000", "East Flanders"),
        ("9999", "East Flanders"),
        ("0000", None),
        (None, None),
    ],
)
def test_convert_zip_to_province(input_value, expected_output):
    assert convert_zip_to_province(input_value) == expected_output


@pytest.mark.parametrize(
    "invalid_input",
    [
        "abcd",
        "12345",
        "-1000",
    ],
)
def test_convert_zip_to_province_raises(invalid_input):
    with pytest.raises(ValueError):
        convert_zip_to_province(invalid_input)


################ test DataCleaner ###############
####################################################


@pytest.fixture
def data_cleaner():
    data = pd.DataFrame(
        {
            "price": ["1,200", "2,300", "3,400"],
            "zip_code": ["1000", "2000", "3000"],
            "energy_class": [" A ", " B ", " C "],
            "primary_energy_consumption": ["1,000", "2,000", "3,000"],
            "bedrooms": ["2", "3", "4"],
            "tenement_building": [" Yes ", " No ", " Yes "],
            "living_area": ["100", "200", "300"],
            "surface_of_the_plot": ["1,000", "2,000", "3,000"],
            "bathrooms": ["1", "2", "3"],
            "double_glazing": [" Yes ", " No ", " Yes "],
            "number_of_frontages": ["2", "3", "4"],
            "building_condition": [" Good ", " Bad ", " Good "],
            "toilets": ["1", "2", "3"],
            "heating_type": [" Gas ", " Electric ", " Gas "],
            "construction_year": ["2000", "2010", "2020"],
        }
    )
    return DataCleaner(data)


def test_add_missing_columns(data_cleaner):
    data = pd.DataFrame(columns=["price", "zip_code"])
    data_with_missing_columns = data_cleaner.add_missing_columns(data)
    assert set(data_cleaner.columns_to_keep).issubset(data_with_missing_columns.columns)


def test_filter_columns(data_cleaner):
    data = pd.DataFrame(columns=data_cleaner.columns_to_keep + ["extra"])
    filtered_data = data_cleaner.filter_columns(data)
    assert set(filtered_data.columns) == set(data_cleaner.columns_to_keep)


def test_reformat_entries(data_cleaner):
    reformatted_data = data_cleaner.reformat_entries(data_cleaner.original_data)
    assert isinstance(reformatted_data, dict)
