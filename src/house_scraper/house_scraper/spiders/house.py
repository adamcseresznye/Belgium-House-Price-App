from datetime import date

import scrapy
from scrapy.loader import ItemLoader

from ..items import HouseScraperItem


class HouseSpider(scrapy.spiders.CrawlSpider):
    name = "house"
    allowed_domains = ["books.toscrape.com"]
    start_urls = ["https://books.toscrape.com/catalogue/page-1.html"]

    rules = (
        # Extract links matching 'page'
        # and follow links from them (since no callback means follow=True by default).
        scrapy.spiders.Rule(scrapy.linkextractors.LinkExtractor(allow=(r"page",))),
        scrapy.spiders.Rule(
            scrapy.linkextractors.LinkExtractor(allow=(r"index",)),
            callback="parse_item",
        ),
    )

    def parse_item(self, response):
        l = ItemLoader(item=HouseScraperItem(), response=response)
        l.add_css("title", "h1")
        l.add_css("price", "p.price_color")
        l.add_value("date", str(date.today()))

        return l.load_item()

    def parse(self, response):
        pass
