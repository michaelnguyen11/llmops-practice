from abc import ABC, abstractmethod
from typing import Any
from langchain.prompts import PromptTemplate
from pydantic import BaseModel

from llm_twin.domain.queries import Query


class PromptTemplateFactory(ABC, BaseModel):
    @abstractmethod
    def create_template(self) -> PromptTemplate:
        pass


# the interface to standardize the interface of advanced RAG steps : query expansion and self-querying
class RAGStep(ABC):
    def __init__(self, mock: bool = False) -> None:
        # use mock attribute to reduce costs and debugging time during development
        self._mock = mock

    @abstractmethod
    def generate(self, query: Query, *args, **kwargs) -> Any:
        pass
