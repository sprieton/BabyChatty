import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import trafilatura
import pandas as pd

from utils import GenConfig as cfg, ScrapingConfig as scfg


class KidsHealthEnSpider(CrawlSpider):

    def __init__(self, *args, **kwargs):
        super(KidsHealthEnSpider, self).__init__(*args, **kwargs)

        # items collected during the crawl will be stored in this list, and processed at the end of the crawl
        self.name = scfg.name
        self.allowed_domains = scfg.allowed_domains

        # entry point: pediatric section in English
        self.start_urls = scfg.start_urls

        self.rules = (
            Rule(
                LinkExtractor(
                    allow_domains=self.allowed_domains,
                    allow=[r'/en/parents/'],
                    deny=[
                        r'/en/kids/all-categories',
                        r'/en/kids/word-',
                        r'/search',
                        r'/es/',          # block all Spanish sections
                        r'/en/kids/',
                        r'/en/teens/',
                    ],
                ),
                callback='parse_article',
                follow=True,
            ),
        )

        self.custom_settings = {
            'DEPTH_LIMIT': scfg.depth_limit,
            'DEPTH_PRIORITY': 1,
            'CONCURRENT_REQUESTS': scfg.concurrent_requests,
            'CLOSESPIDER_ITEMCOUNT': scfg.max_articles,
            'ROBOTSTXT_OBEY': False,
            'DOWNLOAD_DELAY': 0.5,   # small delay to avoid bans
            'USER_AGENT': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0 Safari/537.36'
            ),
            'LOG_LEVEL': 'WARNING',
        }

    @classmethod
    def from_crawler(cls, crawler):
        spider = super().from_crawler(crawler)
        crawler.signals.connect(
            spider.spider_closed,
            signal=scrapy.signals.spider_closed
        )
        return spider

    def spider_closed(self, spider):
        df = pd.DataFrame(self.collected_items)
        # Remove duplicates by URL. avoid spider visit same page twice
        df = df.drop_duplicates(subset='url').reset_index(drop=True)

        print(f'\n✅ Scraping completed: {len(df)} articles recovered.\n')
        print(df[['title', 'url', 'num_keywords']].to_string())

        # Save as a parquet file (usefful to pandas) and a JSON lines file.
        parquet_path = scfg.parquet_file
        jsonl_path = scfg.jsonl_path

        # Save the files
        df.to_parquet(parquet_path, index=False)
        df.to_json(jsonl_path, orient='records', lines=True, force_ascii=False)
        print(f'\n💾 Saved on {parquet_path} and {jsonl_path}')

    def parse_article(self, response):
        # Avoid process non-HTML content (e.g., PDFs, images) - we only want HTML pages
        if not response.url.endswith('.html'):
            return

        # Stract the main text content using trafilatura, which is good for boilerplate removal
        text = trafilatura.extract(
            response.body,
            favor_precision=True,
            include_comments=False,
            include_tables=False,
            deduplicate=True,
        )

        # Discard articles that are too short or don't contain any of the keywords 
        # (to focus on relevant content)
        if not text or len(text.strip()) < scfg.min_length:
            return

        # Keyword filter, at least one should appear in the article
        matched = [kw for kw in scfg.keywords if kw in text.lower()]
        if not matched:
            return

        item = {
            'url': response.url,
            'title': response.css('title::text').get('').strip(),
            'matched_keywords': matched,
            'num_keywords': len(matched),
            'text': text,
        }

        # add the item to the collected items list, it will be processed at the end of the crawl
        self.collected_items.append(item)
        yield item