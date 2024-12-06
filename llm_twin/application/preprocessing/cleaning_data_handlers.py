from abc import ABC
from typing import Generic, TypeVar

from llm_twin.domain.cleaned_documents import (
    CleanedDocument,
    CleanedArticleDocument,
    CleanedPostDocument,
    CleanedRepositoryDocument,
)

from llm_twin.domain.documents import (
    Document,
    ArticleDocument,
    PostDocument,
    RepositoryDocument,
)

