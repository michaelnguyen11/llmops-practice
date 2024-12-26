from langchain_openai import ChatOpenAI
from loguru import logger

from llm_twin.domain.queries import Query
from llm_twin.settings import settings

from .base import RAGStep
from .prompt_templates import QueryExpansionTemplate


class QueryExpansion(RAGStep):
    """
    Generate expanded query versions
    """

    def generate(self, query: Query, expand_to_n: int) -> list[Query]:
        assert (
            expand_to_n > 0
        ), f"'expand_to_n' should be greater than 0. Got {expand_to_n}."

        if self._mock:
            # return a list containing copies of the original query to simulate expansion without calling the API
            return [query for _ in range(expand_to_n)]

        # Create the prompt and initialize the language model
        query_expansion_template = QueryExpansionTemplate()
        prompt = query_expansion_template.create_template(
            expand_to_n - 1
        )  # excluding the original
        model = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )

        # piping the prompt into the model
        chain = prompt | model

        # Invoke it with the user's question
        response = chain.invoke({"question": query})
        result = response.content

        # Parse and clean the expanded queries
        queries_content = result.strip().split(query_expansion_template.separator)

        # Starting with a list containing the original query
        queries = [query]
        # Expand query after stripping any extra whitespace
        queries += [
            query.replace_content(stripped_content)
            for content in queries_content
            if (stripped_content := content.strip())
        ]

        return queries


if __name__ == "__main__":
    query = Query.from_str(
        "Write an article about the best types of advanced RAG methods."
    )
    query_expander = QueryExpansion()
    expanded_queries = query_expander.generate(query, expand_to_n=5)
    for expanded_query in expanded_queries:
        logger.info(f"{expanded_query.content}")
