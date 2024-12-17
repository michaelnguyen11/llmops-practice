import re


# For simplicity, use the same cleaning technique for all the data categories.
# TODO: optimize and create a different cleaning function for each data category.
def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\s.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()
