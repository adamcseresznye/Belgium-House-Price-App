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
    zip_code = scrapy.Field(
        input_processor=MapCompose(remove_tags, extract_numbers),
        output_processor=TakeFirst(),
    )
    date = scrapy.Field(
        output_processor=TakeFirst(),
    )

    def __setitem__(self, key, value):
        # If the field is not predefined, define it
        if key not in self.fields:
            self.fields[key] = scrapy.Field(
                input_processor=MapCompose(remove_tags),
                output_processor=Compose(TakeFirst(), unpack_list),
            )
        super().__setitem__(key, value)
