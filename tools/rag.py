from langchain.globals import set_verbose
from loguru import logger

from llm_twin.application.rag.retriever import ContextRetriever
from llm_twin.domain.embedded_chunks import EmbeddedChunk

if __name__ == "__main__":
    set_verbose(True)

    query = """
        My name is Hiep Nguyen.
        Could you draft a LinkedIn post discussing RAG systems?
        I'm particularly interested in:
            - how RAG works
            - how it is integrated with vector DBs and large language models (LLMs).
        """

    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query, k=4)
    context = EmbeddedChunk.to_context(documents)

    logger.info("Retrieved documents:")
    for rank, document in enumerate(documents):
        logger.info(f"{rank + 1}: {document}")

    logger.info(f"Context : {context}")

