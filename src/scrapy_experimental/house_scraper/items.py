# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import re

import scrapy
from itemloaders.processors import Compose, MapCompose, TakeFirst
from w3lib.html import remove_tags


class HouseScraperItem(scrapy.Item):
    pass


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


def extract_numbers(value):
    numbers = re.findall(r"\d+", value)
    return numbers[0] if numbers else None


def unpack_list(value):
    return value[0] if value else None


class FlexibleItem(scrapy.Item):
    # Predefined fields
    ad_id = scrapy.Field(
        input_processor=MapCompose(remove_tags, extract_numbers),
        output_processor=TakeFirst(),
    )
    list_price = scrapy.Field(
        input_processor=MapCompose(remove_tags, extract_numbers),
        output_processor=TakeFirst(),
    )
    province = scrapy.Field(
        input_processor=MapCompose(
            remove_tags, extract_numbers, convert_zip_to_province
        ),
        output_processor=TakeFirst(),
    )
    date = scrapy.Field(
        output_processor=TakeFirst(),
    )

    def __setitem__(self, key, value):
        # Format the key
        key = key.lower().replace(" ", "_")
        # If the field is not predefined, define it
        if key not in self.fields:
            self.fields[key] = scrapy.Field(
                input_processor=MapCompose(remove_tags),
                output_processor=Compose(TakeFirst(), unpack_list),
            )
        super().__setitem__(key, value)
