from pydantic import UUID4, Field

from llm_twin.domain.base import VectorBaseDocument
from llm_twin.domain.types import DataCategory


class Query(VectorBaseDocument):
    content: str  # contains input query
    author_id: UUID4 | None = (
        None  # A filter within the vector search operation to retrieve chunks written only by a specific author
    )
    author_full_name: str | None = None  # USed to query the author_id
    metadata: dict = Field(default_factory=dict)  # A dict for any additional metadata

    class Config:
        category = DataCategory.QUERIES

    # Create a Query instance directly from a string
    @classmethod
    def from_str(cls, query: str) -> "Query":
        return Query(content=query.strip("\n "))

    # Create a new Query instance with updated content
    def replace_content(self, new_content: str) -> "Query":
        return Query(
            id=self.id,
            content=new_content,
            author_id=self.author_id,
            author_full_name=self.author_full_name,
            metadata=self.metadata,
        )


class EmbeddedQuery(Query):
    """
    Encapsulate all the data and metadata to perform vector search operations on top of vectorDB
    """

    embedding: list[float]

    class Config:
        category = DataCategory.QUERIES
