import re

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from llm_twin.application.networks import EmbeddingModelSingleton

embedding_model = EmbeddingModelSingleton()


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    # Split text based on a given separator or chunk size
    # Using the separator, we first try to find paragraphs in the given text
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n"], chunk_size=chunk_size, chunk_overlap=0
    )
    # But if there are no paragraphs or they are too long, we cut it at a given chunk size
    text_split_by_characters = character_splitter.split_text(text)

    # Ensure that the chunk doesn't exceed the maximum input length of the embedding model.
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap,
        tokens_per_chunk=embedding_model.max_input_length,
        model_name=embedding_model.model_id,
    )

    chunks_by_tokens = []
    for section in text_split_by_characters:
        chunks_by_tokens.extend(token_splitter.split_text(section))

    return chunks_by_tokens


def chunk_document(text: str, min_length: int, max_length: int) -> list[str]:
    pass


def chunk_article(text: str, min_length: int, max_length: int) -> list[str]:
    # Find all the sentences within the given text by looking for periods,question marks or exclamation point followed by a space.
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)

    extracts = []
    current_chunk = ""

    # Groups sentences into a single chunk until  the max_length limi is reached.
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            if len(current_chunk) >= min_length:
                extracts.append(current_chunk.strip())

            current_chunk = sentence + " "

    if len(current_chunk) >= min_length:
        extracts.append(current_chunk.strip())

    return extracts
