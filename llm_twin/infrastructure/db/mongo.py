from loguru import logger
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from llm_twin.settings import settings


class MongoDatabaseConnector:
    _instance: MongoClient | None = None

    def __new__(cls, *args, **kwargs) -> MongoClient:
        if cls._instance is None:
            try:
                cls._instance = MongoClient(settings.DATABASE_HOST)
            except ConnectionFailure as e:
                logger.error("Couldn't connect to the database: {}".format(e))
                raise

        logger.info(
            "Connection to MongoDB with URI successful: {}".format(
                settings.DATABASE_HOST
            )
        )

        return cls._instance


connection = MongoDatabaseConnector()
