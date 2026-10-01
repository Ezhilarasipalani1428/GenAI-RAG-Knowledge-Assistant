"""
embeddings.py

Converts text into embedding vectors using Google's Gemini embedding
model (gemini-embedding-001) through the current `google-genai` SDK.
This is the third step of our RAG pipeline.

What is an embedding?
An embedding is a list of numbers (a vector) that represents the MEANING
of a piece of text. Texts with similar meanings get vectors that are close
to each other, so we can find relevant chunks by comparing vectors instead
of matching exact keywords.
"""

import os

from dotenv import load_dotenv
from google import genai

# The Gemini model that turns text into vectors
EMBEDDING_MODEL = "gemini-embedding-001"

# The API accepts a limited number of texts per request,
# so we send large lists in batches of this size
BATCH_SIZE = 100

# Load variables from the .env file into the environment.
# The API key is NEVER written in the code.
load_dotenv()

# The client is created only when first needed (see _get_client)
_client = None


def _get_client():
    """
    Create the Gemini client once and reuse it.

    Raises:
        ValueError: If no API key is found in the environment / .env file.
    """
    global _client

    if _client is None:
        # Look for the key under the common variable names
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "API key not found. Add GEMINI_API_KEY=your_key to your "
                ".env file in the project root."
            )

        _client = genai.Client(api_key=api_key)

    return _client


def generate_embedding(text):
    """
    Generate the embedding vector for ONE piece of text.

    Args:
        text (str): The text to embed.

    Returns:
        list[float]: The embedding vector.

    Raises:
        ValueError: If text is not a string or is empty.
        RuntimeError: If the Gemini API call fails.
    """
    # Validate the input
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    # Reuse generate_embeddings() with a list of one item
    return generate_embeddings([text])[0]


def generate_embeddings(texts):
    """
    Generate embedding vectors for a LIST of texts.

    Args:
        texts (list[str]): The texts to embed (e.g. all chunk texts).

    Returns:
        list[list[float]]: One vector per text, in the same order.

    Raises:
        ValueError: If texts is not a non-empty list of non-empty strings.
        RuntimeError: If the Gemini API call fails.
    """
    # 1. Validate the input
    if not isinstance(texts, list) or len(texts) == 0:
        raise ValueError("texts must be a non-empty list of strings")

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("every item in texts must be a non-empty string")

    client = _get_client()
    all_vectors = []

    # 2. Send the texts to Gemini in batches
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
            )
        except Exception as error:
            raise RuntimeError(f"Embedding request failed: {error}") from error

        # 3. response.embeddings holds one object per text;
        #    its .values attribute is the list of numbers (the vector)
        for embedding in response.embeddings:
            all_vectors.append(list(embedding.values))

    # 4. Safety check: we must get exactly one vector per text
    if len(all_vectors) != len(texts):
        raise RuntimeError(
            f"Expected {len(texts)} embeddings but received {len(all_vectors)}"
        )

    return all_vectors


# Simple test: run from the project root with
#   python src/embeddings.py
if __name__ == "__main__":
    try:
        vector = generate_embedding("What is Retrieval-Augmented Generation?")
        print("Single embedding length:", len(vector))
        print("First 5 numbers:", vector[:5])

        vectors = generate_embeddings(["Hello world", "How are you?"])
        print("Number of embeddings:", len(vectors))
    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}")