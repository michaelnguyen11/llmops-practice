from abc import ABC, abstractmethod
from typing import Generic, TypeVar, cast

from llm_twin.application.networks import EmbeddingModelSingleton
from llm_twin.domain.chunks import Chunk, ArticleChunk, PostChunk, RepositoryChunk
from llm_twin.domain.embedded_chunks import (
    EmbeddedChunk,
    EmbeddedArticleChunk,
    EmbeddedPostChunk,
    EmbeddedRepositoryChunk,
)

ChunkT = TypeVar("ChunkT", bound=Chunk)
EmbeddedChunkT = TypeVar("EmbeddedChunkT", bound=EmbeddedChunk)

embedding_model = EmbeddingModelSingleton()


class EmbeddingDataHandler(ABC, Generic[ChunkT, EmbeddedChunkT]):
    """
    Abstract class for all embedding data handlers.
    All data transformations logic for the embedding step is done here.
    """

    # Run the inference on a single data point
    def embed(self, data_model: ChunkT) -> EmbeddedChunkT:
        return self.embed_batch([data_model])[0]

    # Takes chunked documents as input, gathers their content into a list
    # Then, passes them to the embedding model and maps the results to an embedded chunk domain entity.
    def embed_batch(self, data_models: list[ChunkT]) -> list[EmbeddedChunkT]:
        embedding_model_input = [data_model.content for data_model in data_models]
        embeddings = embedding_model(embedding_model_input, to_list=True)

        embedded_chunk = [
            self.map_model(data_model, cast(list[float], embedding))
            for data_model, embedding in zip(data_models, embeddings, strict=True)
        ]

        return embedded_chunk

    @abstractmethod
    def map_model(self, data_model: ChunkT, embedding: list[float]) -> EmbeddedChunkT:
        pass


class PostEmbeddingHandler(EmbeddingDataHandler):
    def map_model(
        self, data_model: PostChunk, embedding: list[float]
    ) -> EmbeddedPostChunk:
        return EmbeddedPostChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            platform=data_model.platform,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )


class ArticleEmbeddingHandler(EmbeddingDataHandler):
    def map_model(
        self, data_model: ArticleChunk, embedding: list[float]
    ) -> EmbeddedArticleChunk:
        return EmbeddedArticleChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            platform=data_model.platform,
            link=data_model.link,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )


class RepositoryEmbeddingHandler(EmbeddingDataHandler):
    def map_model(
        self, data_model: RepositoryChunk, embedding: list[float]
    ) -> EmbeddedRepositoryChunk:
        return EmbeddedRepositoryChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            platform=data_model.platform,
            name=data_model.name,
            link=data_model.link,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )
