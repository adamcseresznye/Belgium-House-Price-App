# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


import sys

import pymongo

from .items import FlexibleItem, HouseScraperItem


class HouseScraperPipeline:
    def process_item(self, item, spider):
        return item


class MongoDBPipeline:
    collection = "BE_houses"

    def __init__(self, mongodb_uri, mongodb_db):
        self.mongodb_uri = mongodb_uri
        self.mongodb_db = mongodb_db
        if not self.mongodb_uri:
            sys.exit("You need to provide a Connection String.")

    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            mongodb_uri=crawler.settings.get("MONGODB_URI"),
            mongodb_db=crawler.settings.get("MONGODB_DATABASE", "items"),
        )

    def open_spider(self, spider):
        self.client = pymongo.MongoClient(self.mongodb_uri)
        self.db = self.client[self.mongodb_db]
        # Start with a clean database
        # self.db[self.collection].delete_many({})

    def close_spider(self, spider):
        self.client.close()

    # def process_item(self, item, spider):
    #    data = dict(FlexibleItem(item))
    #    self.db[self.collection].insert_one(data)
    #    return item

    def process_item(self, item, spider):
        # Convert item to FlexibleItem
        data = dict(FlexibleItem(item))
        # Unpack lists in data
        for key, value in data.items():
            if isinstance(value, list) and len(value) == 1:
                data[key] = value[0]
        # Insert data into database
        self.db[self.collection].insert_one(data)
        return item