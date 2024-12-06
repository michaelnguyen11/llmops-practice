from InstructorEmbedding import INSTRUCTOR

if __name__ == "__main__":
    model = INSTRUCTOR("hkunlp/instructor-base")

    sentence = "RAG Fundamentals First"

    instruction = "Represent the title of an article about AI:"

    embeddings = model.encode([[instruction, sentence]])
    print(embeddings.shape)  # noqa
    # Output: (1, 768)