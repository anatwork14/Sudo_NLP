import scrapy
import logging

# Set logging level to INFO for more useful output during scraping
logging.basicConfig(level=logging.INFO)

class VnexpressSpider(scrapy.Spider):
    """
    Scrapy Spider designed to crawl the 'The Thao' (Sports) section of VNExpress.
    
    It extracts article titles, descriptions (summaries), and direct links 
    from the main listing page.
    """
    name = 'vnexpress_sports'
    allowed_domains = ['vnexpress.net']
    start_urls = ['https://vnexpress.net/the-thao']

    # --- Custom Settings for better behavior ---
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'LOG_LEVEL': 'INFO',
    }

    def parse(self, response):
        # Check status code
        if response.status != 200:
            self.logger.error(f"Received non-200 status code: {response.status} for URL: {response.url}")
            return
            
        # *** CRITICAL FIX ATTEMPT #3: Force Content-Type header to ensure correct UTF-8 string is used by the parser ***
        # We explicitly set the Content-Type header to include the UTF-8 charset. 
        # This often forces the underlying lxml/parsel parser to correctly recognize the encoding string 
        # and bypasses the bug causing the "unknown encoding: b'utf8'" error.
        response.headers['Content-Type'] = b'text/html; charset=utf-8'
        
        # We also need to explicitly set the encoding of the response object, though this is often redundant 
        # when setting the header, it provides an additional layer of protection.
        response._encoding = 'utf-8' # Using the internal property for maximum effect
        
        # --- Extraction Method: Using CSS Selectors ---
        
        # 1. Strategy: Use a class selector (.item-news) for the main container.
        article_containers = response.css('.item-news')
        
        self.logger.info(f"Found {len(article_containers)} article containers with class '.item-news' on {response.url}")

        if not article_containers:
            self.logger.warning("Primary selector failed. Trying a secondary selector.")
            article_containers = response.css('.list-news-subfolder article')
            self.logger.info(f"Attempted secondary selector, found {len(article_containers)} articles.")
            if not article_containers:
                 self.logger.warning("Secondary selector also failed. Check site structure or connectivity.")
                 return

        # Iterate through each article container found on the page
        for article in article_containers:
            # 2. Extract the data points from each container

            title_link_element = article.css('h3.title-news a')
            
            title = title_link_element.css('::text').get()
            url = title_link_element.css('::attr(href)').get()

            # Description (Summary)
            description_element = article.css('p.description')
            description = description_element.css('a::text').get()
            if not description:
                 direct_text = description_element.css('::text').get()
                 description = direct_text.strip() if direct_text else None
            
            # If the data is valid, yield the structured data.
            if title and url:
                yield {
                    'title': title.strip() if title else None,
                    'description': description if description else 'No summary available.',
                    'url': response.urljoin(url)
                }
        
        # --- Pagination Logic ---
        
        # Look for the next page link
        next_page = response.css('a[data-medium="Next"]::attr(href)').get()

        if next_page is not None:
            self.logger.info(f"Following next page link: {next_page}")
            yield response.follow(next_page, callback=self.parse)