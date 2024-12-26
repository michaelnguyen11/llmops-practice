from langchain_openai import ChatOpenAI
from loguru import logger

from llm_twin.application import utils
from llm_twin.domain.documents import UserDocument
from llm_twin.domain.queries import Query
from llm_twin.settings import settings

from .base import RAGStep
from .prompt_templates import SelfQueryTemplate


class SelfQuery(RAGStep):
    """
    By implementing self-querying, we ensure that critical metadata 
        required for our use case is ex-plicitly extracted and used during retrieval.
    This approach overcomes the limitations of relying 
        solely on the semantics of the embeddings to capture all necessary aspects of a query.
    """

    def generate(self, query: Query) -> Query:
        if self._mock:
            return query

        prompt = SelfQueryTemplate().create_template()
        model = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )

        # Combine the prompt and the model into a chain
        chain = prompt | model

        response = chain.invoke({"question": query})
        user_full_name = response.content.strip("\n ")

        if user_full_name == "none":
            return query

        first_name, last_name = utils.split_user_full_name(user_full_name)
        # Retrieve or create a UserDocument user instance
        user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)

        # Update the query object with the extracted author information
        query.author_id = user.id
        query.author_full_name = user.full_name

        return query


if __name__ == "__main__":
    query = Query.from_str(
        "I am Hiep Nguyen. Write an article about the best types of advanced RAG methods."
    )
    self_query = SelfQuery()
    query = self_query.generate(query)
    logger.info(f"Extracted author_id: {query.author_id}")
    logger.info(f"Extracted author_full_name: {query.author_full_name}")
