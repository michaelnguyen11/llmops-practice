import hashlib
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

from llm_twin.domain.chunks import Chunk, ArticleChunk, PostChunk, RepositoryChunk
from llm_twin.domain.cleaned_documents import (
    CleanedDocument,
    CleanedArticleDocument,
    CleanedPostDocument,
    CleanedRepositoryDocument,
)

from .operations import chunk_article, chunk_text

CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)
ChunkT = TypeVar("ChunkT", bound=Chunk)


# The handler takes cleaned documents as input -> returns chunk entities.
class ChunkingDataHandler(ABC, Generic[CleanedDocumentT, ChunkT]):
    """
    Abstract class for all chunking data handlers.
    All data transformations logic for the chunking step is done here.
    """

    # Property to aggregate all the neccessary properties required for chunking in a single struct.
    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

    @abstractmethod
    def chunk(self, data_model: CleanedDocumentT) -> list[ChunkT]:
        pass


class PostChunkingHandler(ChunkingDataHandler):
    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 250,
            "chunk_overlap": 25,
        }

    def chunk(self, data_model: CleanedPostDocument) -> list[PostChunk]:
        data_models_list = []

        cleaned_content = data_model.content
        chunks = chunk_text(
            cleaned_content,
            chunk_size=self.metadata["chunk_size"],
            chunk_overlap=self.metadata["chunk_overlap"],
        )

        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = PostChunk(
                id=UUID(chunk_id, version=4),
                content=chunk,
                platform=data_model.platform,
                document_id=data_model.id,
                author_id=data_model.author_id,
                author_full_name=data_model.author_full_name,
                image=data_model.image if data_model.image else None,
                metadata=self.metadata,
            )
            data_models_list.append(model)

        return data_models_list


class ArticleChunkingHandler(ChunkingDataHandler):
    @property
    def metadata(self) -> dict:
        return {
            "min_length": 1000,
            "max_length": 2000,
        }

    # Input cleaned article documents and returns a list of article chunk entities.
    def chunk(self, data_model: CleanedArticleDocument) -> list[ArticleChunk]:
        data_models_list = []

        cleaned_content = data_model.content
        chunks = chunk_article(
            cleaned_content,
            min_length=self.metadata["min_length"],
            max_length=self.metadata["max_length"],
        )

        for chunk in chunks:
            # MD5 hash of chunk's content
            # Thus, if the two chunks have precisely the same content
            # They will have the same ID -> we can easily deduplicate them
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = ArticleChunk(
                id=UUID(chunk_id, version=4),
                content=chunk,
                platform=data_model.platform,
                link=data_model.link,
                document_id=data_model.id,
                author_id=data_model.author_id,
                author_full_name=data_model.author_full_name,
                metadata=self.metadata,
            )
            data_models_list.append(model)

        return data_models_list


class RepositoryChunkingHandler(ChunkingDataHandler):
    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 1500,
            "chunk_overlap": 100,
        }

    def chunk(self, data_model: CleanedRepositoryDocument) -> list[RepositoryChunk]:
        data_models_list = []

        cleaned_content = data_model.content
        chunks = chunk_text(
            cleaned_content,
            chunk_size=self.metadata["chunk_size"],
            chunk_overlap=self.metadata["chunk_overlap"],
        )

        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = RepositoryChunk(
                id=UUID(chunk_id, version=4),
                content=chunk,
                platform=data_model.platform,
                name=data_model.name,
                link=data_model.link,
                document_id=data_model.id,
                author_id=data_model.author_id,
                author_full_name=data_model.author_full_name,
                metadata=self.metadata,
            )
            data_models_list.append(model)

        return data_models_list
