import concurrent.futures

from loguru import logger
from qdrant_client.models import FieldCondition, Filter, MatchValue

from llm_twin.application import utils
from llm_twin.application.preprocessing.dispatchers import EmbeddingDispatcher
from llm_twin.domain.embedded_chunks import (
    EmbeddedChunk,
    EmbeddedArticleChunk,
    EmbeddedPostChunk,
    EmbeddedRepositoryChunk,
)

from llm_twin.domain.queries import Query, EmbeddedQuery

from .query_expansion import QueryExpansion
from .self_query import SelfQuery
from .reranking import Reranker


class ContextRetriever:
    def __init__(self, mock: bool = False) -> None:
        self._query_expander = QueryExpansion(mock=mock)
        self._metadata_extractor = SelfQuery(mock=mock)
        self._reranker = Reranker(mock=mock)

    def search(self, query: Query, k: int = 4, expand_to_n_queries: int = 3) -> list:
        # Convert the user's input string into a Query object
        query_model = Query.from_str(query=query)
        # Extract the metadata from the query
        query_model = self._metadata_extractor.generate(query_model)

        if query_model.author_full_name is not None:
            logger.info(
                f"Successfully extracted the author_full_name = {query_model.author_full_name} from the query.",
            )

        # Expand the query to generate multiple semantically similar queries
        n_generated_queries = self._query_expander.generate(
            query_model, expand_to_n=expand_to_n_queries
        )
        logger.info(
            f"Successfully generated {len(n_generated_queries)} search queries.",
        )

        # Perform the search concurrently for all expanded queries using a thread pool
        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_tasks = [
                executor.submit(self._search, _query_model, k)
                for _query_model in n_generated_queries
            ]

            n_k_documents = [
                task.result() for task in concurrent.futures.as_completed(search_tasks)
            ]
            n_k_documents = utils.misc.flatten(n_k_documents)  # flatten a list of lists
            n_k_documents = list(set(n_k_documents))  # deduplicated chunks

        logger.info(f"{len(n_k_documents)} documents retrieved successfully")

        # Perform reranking and keep only top K

        if len(n_k_documents) > 0:
            k_documents = self.rerank(query, chunks=n_k_documents, keep_top_k=k)
        else:
            k_documents = []

        return k_documents

    def _search(self, query: Query, k: int = 4) -> list[EmbeddedChunk]:
        assert k >= 2, "k should be >= 2"

        def _search_data_category(
            data_category_odm: type[EmbeddedChunk], embedded_query: EmbeddedQuery
        ) -> list[EmbeddedChunk]:
            if embedded_query.author_id:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="author_id",
                            match=MatchValue(
                                value=str(embedded_query.author_id),
                            ),
                        )
                    ]
                )
            else:
                query_filter = None

            return data_category_odm.search(
                query_vector=embedded_query.embedding,
                limit=k // 2,
                query_filter=query_filter,
            )

        # Convert the query into an EmbeddedQuery using EmbeddingDispatcher
        embedded_query: EmbeddedQuery = EmbeddingDispatcher.dispatch(query)

        # post_chunks = _search_data_category(EmbeddedPostChunk, embedded_query)
        articles_chunks = _search_data_category(EmbeddedArticleChunk, embedded_query)
        repositories_chunks = _search_data_category(
            EmbeddedRepositoryChunk, embedded_query
        )
        retrieved_chunks = articles_chunks + repositories_chunks

        return retrieved_chunks

    def rerank(
        self, query: str | Query, chunks: list[EmbeddedChunk], keep_top_k: int
    ) -> list[EmbeddedChunk]:
        if isinstance(query, str):
            query = Query.from_str(query=query)

        reranked_documents = self._reranker.generate(
            query=query, chunks=chunks, keep_top_k=keep_top_k
        )
        logger.info(f"{len(reranked_documents)} documents reranked successfully.")

        return reranked_documents
