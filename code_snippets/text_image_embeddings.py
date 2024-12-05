from io import BytesIO

import requests
from PIL import Image
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    # Load an image with a crazy cat
    response = requests.get(
        "https://github.com/PacktPublishing/LLM-Engineering/blob/main/images/crazy_cat.jpg?raw=true"
    )
    image = Image.open(BytesIO(response.content))
    print("image shape : {}".format(image.size))
    # image shape : (640, 960)

    # Load CLIP model
    model = SentenceTransformer("clip-ViT-B-32")

    # Encode the loaded image
    image_embedding = model.encode(image)

    print(f"Image embedding shape: {image_embedding.shape}")
    # Image embedding shape: (512,)

    # Load a text description of the cat
    text_description = [
        "A crazy cat smiling.",
        "A white and brown cat with a yellow bandana.",
        "A man eating in the garden.",
    ]

    # Encode the text description
    text_embedding = model.encode(text_description)
    print(f"Text embedding shape: {text_embedding.shape}")
    # Text embedding shape: (3, 512)

    # Compute similarities
    similarity_scores = model.similarity(image_embedding, text_embedding)
    print(f"Similarity scores: {similarity_scores}")
    # Similarity scores: tensor([[0.3068, 0.3300, 0.1719]])
