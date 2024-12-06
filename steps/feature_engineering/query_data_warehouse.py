from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_twin.application import utils
from llm_twin.domain.base.nosql import NoSQLBaseDocument
from llm_twin.domain.documents import (
    ArticleDocument,
    Document,
    RepositoryDocument,
    UserDocument,
)


@step
def query_data_warehouse(
    author_full_names: list[str],
) -> Annotated[list, "raw_documents"]:
    documents = []
    authors = []

    for author_full_name in author_full_names:
        logger.info(f"Querying data warehouse for user: {author_full_name}")
        # Attempts to get or create a UserDocument instance using the first name and last name.
        # if user does not exist, it throws an error
        first_name, last_name = utils.split_user_full_name(author_full_name)
        logger.info(f"First name: {first_name}, Last name: {last_name}")
        user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)
        authors.append(user)

        # Fetches all the raw data for the user from the data warehouse
        results = fetch_all_data(user=user)

        # Extends the documents list to include these user documents
        user_documents = [
            doc for query_result in results.values() for doc in query_result
        ]
        documents.extend(user_documents)

    # Compute a descriptive metadata dictionary in ZenML
    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="raw_documents", metadata=__get_metadata(documents)
    )

    return documents


def fetch_all_data(user: UserDocument) -> dict[str, list[NoSQLBaseDocument]]:
    user_id = str(user.id)
    with ThreadPoolExecutor as executor:
        # query on a different thread
        future_to_query = {
            executor.submit(__fetch_articles, user_id): "articles",
            executor.submit(__fetch_repositories, user_id): "repositories",
        }

        results = {}
        for future in as_completed(future_to_query):
            query_name = future_to_query[future]
            try:
                results[query_name] = future.result()
            except Exception as e:
                logger.error(f"'{query_name}' request failed: {e}")
                results[query_name] = []

    return results


def __fetch_articles(user_id: str) -> list[NoSQLBaseDocument]:
    return ArticleDocument.bulk_find(author_id=user_id)


def __fetch_repositories(user_id: str) -> list[NoSQLBaseDocument]:
    return RepositoryDocument.bulk_find(author_id=user_id)


def __get_metadata(documents: list[Document]) -> dict:
    metadata = {
        "num_documents": len(documents),
    }

    for document in documents:
        collection = document.get_collection_name()
        if collection not in metadata:
            metadata[collection] = {}
        if "authors" not in metadata[collection]:
            metadata[collection]["authors"] = list()

        metadata[collection]["num_documents"] = (
            metadata[collection].get("num_documents", 0) + 1
        )
        metadata[collection]["authors"].append(document.author_full_name)

    for value in metadata.values():
        if isinstance(value, dict) and "authors" in value:
            value["authors"] = list(set(value["authors"]))

    return metadata
