"""
vector_store.py

Stores chunk embeddings in a local ChromaDB database and searches them.
This is the fourth step of our RAG pipeline.

What is a vector database?
A normal database finds rows by exact values (like a name or an ID).
A vector database stores embedding vectors and finds the ones that are
CLOSEST in meaning to a query vector.

What does ChromaDB store?
For every chunk, ChromaDB stores four things:
  1. an ID        (our chunk_id, e.g. "error.pdf_p1_c1")
  2. an embedding (the vector used for searching)
  3. a document   (the original chunk text)
  4. metadata     (source file, page number, chunk_id)

Why store the text and metadata together with the embedding?
The vector is only used to FIND the chunk. After the search we need the
original text to give to the LLM as context, and the metadata (file name
and page number) to show where the answer came from.

This file makes NO API calls. Embeddings are created in embeddings.py.
"""

import chromadb


class VectorStore:
    """A small wrapper around a persistent ChromaDB collection."""

    def __init__(self, persist_directory="data/chroma", collection_name="documents"):
        """
        Open (or create) the local database and the collection.

        Args:
            persist_directory (str): Folder where the database files are saved.
                Run the program from the project root so that
                "data/chroma" is created inside the project.
            collection_name (str): Name of the collection (like a table).
        """
        self.collection_name = collection_name

        # PersistentClient saves everything to disk, so the data is still
        # there the next time the program runs.
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get the collection if it exists, otherwise create it
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(self, chunks, embeddings):
        """
        Store chunks and their embeddings in the database.

        Args:
            chunks (list[dict]): Output of chunk_documents().
            embeddings (list[list[float]]): Output of generate_embeddings(),
                one vector per chunk, in the same order.

        Returns:
            int: Number of chunks stored.

        Raises:
            ValueError: If the number of chunks and embeddings differ.
        """
        # 1. Validate: every chunk needs exactly one embedding
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) and embeddings "
                f"({len(embeddings)}) must be equal"
            )

        # Nothing to store
        if len(chunks) == 0:
            return 0

        # 2. Split the chunk dictionaries into the separate lists
        #    that ChromaDB expects
        ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        # 3. Save everything. "upsert" means: insert new items and update
        #    items whose ID already exists, so adding the same PDF twice
        #    does not cause a duplicate-ID error.
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(self, query_embedding, top_k=5):
        """
        Find the chunks most similar to the query embedding.

        Similarity search compares the query vector with the stored vectors
        and returns the ones that are closest, i.e. the chunks whose meaning
        is most similar to the question.

        Args:
            query_embedding (list[float]): Embedding of the user's question.
            top_k (int): How many results to return.

        Returns:
            list[dict]: Best matches first. Each item looks like:
                {
                    "id": "error.pdf_p1_c1",
                    "text": "chunk text...",
                    "metadata": {"source": "error.pdf", "page": 1, "chunk_id": "..."},
                    "distance": 0.42   # smaller = more similar
                }
            An empty list is returned if the database is empty.

        Raises:
            ValueError: If top_k is not a positive integer.
        """
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        # Empty collection: nothing to search
        total = self.count()
        if total == 0:
            return []

        # We cannot ask for more results than there are stored chunks
        n_results = min(top_k, total)

        # ChromaDB returns lists of lists (one inner list per query).
        # We send only one query, so we read index [0] of each list.
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        matches = []
        for i in range(len(results["ids"][0])):
            matches.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return matches

    def count(self):
        """Return the number of chunks stored in the collection."""
        return self.collection.count()

    def clear(self):
        """Remove all stored documents from the collection."""
        # Deleting the collection and creating it again gives an empty
        # collection with the same name.
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )


# Simple test (no API calls, uses tiny fake 3-number vectors and a
# separate test folder so your real database is not touched):
#   python src/vector_store.py
if __name__ == "__main__":
    store = VectorStore(persist_directory="data/chroma_test", collection_name="test")
    store.clear()

    demo_chunks = [
        {"text": "Cats are small animals.",
         "metadata": {"source": "demo.pdf", "page": 1, "chunk_id": "demo.pdf_p1_c1"}},
        {"text": "Python is a programming language.",
         "metadata": {"source": "demo.pdf", "page": 2, "chunk_id": "demo.pdf_p2_c1"}},
    ]
    demo_embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    store.add_documents(demo_chunks, demo_embeddings)
    print("Stored chunks:", store.count())

    # This query vector is closest to the first chunk
    for match in store.search([0.9, 0.1, 0.0], top_k=2):
        print(match["metadata"], "-> distance:", round(match["distance"], 3))

    store.clear()
    print("After clear:", store.count())
    print("Search on empty store:", store.search([0.9, 0.1, 0.0]))