import sys
import os
sys.path.append(os.getcwd() + "/..")

from urllib.parse import urlparse
from loguru import logger

from llm_twin.application.crawlers.dispatcher import CrawlerDispatcher
from llm_twin.domain.documents import UserDocument
from llm_twin.application import utils


user_full_name = "Paul Iusztin"
link = "https://decodingml.substack.com/p/real-time-feature-pipelines-with?r=1ttoeh"

if __name__ == "__main__":

    first_name, last_name = utils.split_user_full_name(user_full_name)
    logger.info(f"First name: {first_name}, Last name: {last_name}")
    user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)

    dispatcher = CrawlerDispatcher.build().register_medium().register_github()

    crawler = dispatcher.get_crawler(link)
    crawler_domain = urlparse(link).netloc
    logger.info(f"Crawling domain: {crawler_domain}")

    try:
        crawler.extract(link=link, user=user)

    except Exception as e:
        logger.error(f"An error occurred while crawling: {e!s}")
