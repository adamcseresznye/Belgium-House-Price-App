from datetime import date

import scrapy
from house_scraper.items import FlexibleItem, HouseScraperItem
from scrapy.loader import ItemLoader
from scrapy_selenium import SeleniumRequest
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC


class HouseSpider(scrapy.Spider):
    name = "house"

    def start_requests(self):
        url = "https://www.immoweb.be/en/search/house/for-sale?countries=BE&page=332&orderBy=relevance"
        yield SeleniumRequest(
            url=url,
            callback=self.parse_links,
            wait_time=1000,
            wait_until=EC.element_to_be_clickable((By.CLASS_NAME, "grid")),
        )

    def parse_links(self, response):
        # Extract the links to the individual ads on the page
        links = response.css(
            "h2.card__title.card--result__title a::attr(href)"
        ).getall()

        for link in links:
            yield SeleniumRequest(url=link, callback=self.parse_ad)

        # Handle pagination
        next_page = response.css("a.pagination__link--next::attr(href)").get()

        if next_page is not None:
            yield SeleniumRequest(url=next_page, callback=self.parse_links)

    def parse_ad(self, response):
        l = ItemLoader(item=FlexibleItem(), response=response)

        # Extract the data from the individual ad that are in tables
        table = response.css("table.classified-table")
        rows = table.css("tr")
        for row in rows:
            header = row.css("th::text").get()
            value = row.css("td::text").get()
            if header and value:
                header = header.strip()
                value = value.strip()
                if header not in l.item.fields:
                    l.item.fields[header] = scrapy.Field()
                l.add_value(header, value)

        # Extract information not in tables:
        l.add_css("list_price", "p.classified__price span.sr-only::text")
        l.add_css(
            "province", "span.classified__information--address-row:nth-child(2)::text"
        )
        l.add_css("ad_id", "div.classified__header--immoweb-code::text")
        l.add_value("date", str(date.today()))

        return l.load_item()
