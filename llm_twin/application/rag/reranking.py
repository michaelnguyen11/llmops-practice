from llm_twin.application.networks import CrossEncoderModelSingleton
from llm_twin.domain.embedded_chunks import EmbeddedChunk
from llm_twin.domain.queries import Query

from .base import RAGStep


class Reranker(RAGStep):
    """
    The class is responsible for reranking the retrieved documents
    based on their relevance to the query.
    """

    def __init__(self, mock: bool = False) -> None:
        super().__init__(mock)

        # The model used to score the relevance of each document chunk with respect to the query
        self._model = CrossEncoderModelSingleton()

    def generate(
        self, query: Query, chunks: list[EmbeddedChunk], keep_top_k: int
    ) -> list[EmbeddedChunk]:
        if self._mock:
            return chunks

        # Create pairs of the query content and each chunk's content
        query_doc_tuples = [(query.content, chunk.content) for chunk in chunks]
        # Use the cross-encoder model to score each pair, assessing how well the chunk matches
        scores = self._model(query_doc_tuples)

        # Create a scored list of tuples and sort based on the scores
        scored_query_doc_tuples = list(zip(scores, chunks, strict=False))
        scored_query_doc_tuples.sort(key=lambda x: x[0], reverse=True)

        reranked_documents = scored_query_doc_tuples[:keep_top_k]
        # Extract the chunks from the tuples
        reranked_documents = [doc for _, doc in reranked_documents]

        return reranked_documents
