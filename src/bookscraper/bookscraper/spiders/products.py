import scrapy
from scrapy.loader import ItemLoader

from ..items import BookscraperItem


class ProductsSpider(scrapy.spiders.CrawlSpider):
    name = "products"
    allowed_domains = ["books.toscrape.com"]
    start_urls = ["https://books.toscrape.com"]

    def parse(self, response):
        pass


class MySpider(scrapy.spiders.CrawlSpider):
    name = "products"
    allowed_domains = ["books.toscrape.com"]
    start_urls = ["https://books.toscrape.com/catalogue/page-1.html"]

    rules = (
        # Extract links matching 'page'
        # and follow links from them (since no callback means follow=True by default).
        scrapy.spiders.Rule(scrapy.linkextractors.LinkExtractor(allow=(r"page",))),
        scrapy.spiders.Rule(
            scrapy.linkextractors.LinkExtractor(allow=(r"catalogue",)),
            callback="parse_item",
        ),
    )

    def parse_item(self, response):
        l = ItemLoader(item=BookscraperItem(), response=response)
        l.add_css("title", "h1")
        l.add_css("price", "p.price_color")

        return l.load_item()
