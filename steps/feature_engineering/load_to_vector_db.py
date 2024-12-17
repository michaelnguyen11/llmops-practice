from loguru import logger
from typing_extensions import Annotated
from zenml import step

from llm_twin.application import utils
from llm_twin.domain.base import VectorBaseDocument


@step
def load_to_vector_db(
    documents: Annotated[list, "documents"],
) -> Annotated[bool, "successful"]:
    """
    As each article, post or code repository sits in a different collection inside the vector DB.
    We have to group all the documents based on their data category.
    Then, we load each group in bulk in the Qdrant vector DB.
    """
    logger.info(f"Loading {len(documents)} documents into the vector database.")

    grouped_documents = VectorBaseDocument.group_by_class(documents=documents)
    for document_class, documents in grouped_documents.items():
        logger.info(f"Loading documents into {document_class.get_collection_name()}")
        for document_batch in utils.misc.batch(documents, size=4):
            try:
                document_class.bulk_insert(document_batch)
            except Exception:
                logger.error(
                    f"Failed to insert documents into {document_class.get_collection_name()}"
                )
                return False

    return True
