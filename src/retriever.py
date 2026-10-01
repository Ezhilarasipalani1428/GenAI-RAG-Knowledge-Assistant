"""
retriever.py

The RETRIEVAL step of RAG (Retrieval-Augmented Generation).

Given a user's question, this module finds the chunks of the uploaded
documents that are most relevant to it:
  1. Convert the question into an embedding vector (embeddings.py).
  2. Compare that vector with the stored chunk vectors (vector_store.py).
  3. Return the closest chunks.

These chunks will later be given to the LLM as context, so that its answer
is based on our documents instead of only on its own memory.
"""

import sys

# Works both when imported from the project root (src.retriever)
# and when this file is run directly (python src/retriever.py)
try:
    from src.embeddings import generate_embedding
    from src.vector_store import VectorStore
except ImportError:
    from embeddings import generate_embedding
    from vector_store import VectorStore


def retrieve(query, top_k=5, store=None):
    """
    Retrieve the chunks most relevant to a user's question.

    Args:
        query (str): The user's question.
        top_k (int): How many chunks to return.
        store (VectorStore, optional): The vector store to search.
            If not supplied, the default VectorStore() is used.

    Returns:
        list[dict]: Best matches first. Each item looks like:
            {
                "id": "error.pdf_p1_c1",
                "text": "chunk text...",
                "metadata": {"source": "error.pdf", "page": 1, "chunk_id": "..."},
                "distance": 0.42   # smaller = more similar
            }
        An empty list is returned if the vector store is empty.

    Raises:
        ValueError: If the query is empty or top_k is invalid.
        RuntimeError: If the embedding request fails.
    """
    # 1. Validate the query
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    # 2. Use the default vector store if none was given
    if store is None:
        store = VectorStore()

    # 3. Turn the question into an embedding vector.
    #    It must be made with the same model as the stored chunks,
    #    otherwise the vectors cannot be compared.
    query_embedding = generate_embedding(query.strip())

    # 4. Find the closest chunks in the vector store
    return store.search(query_embedding, top_k=top_k)


# Simple test: run from the project root with
#   python src/retriever.py "your question here"
# (it searches the chunks already stored in data/chroma)
if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is this document about?"

    try:
        results = retrieve(question, top_k=3)
    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        sys.exit(1)

    print(f"Question: {question}")

    if not results:
        print("No results. The vector store is empty - add documents first.")
    else:
        for rank, match in enumerate(results, start=1):
            print("-" * 40)
            print(f"Result {rank} (distance: {match['distance']:.4f})")
            print("Source:", match["metadata"]["source"],
                  "| Page:", match["metadata"]["page"])
            print("Text preview:", match["text"][:200])