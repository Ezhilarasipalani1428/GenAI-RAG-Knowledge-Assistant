"""
ingest.py

The INGESTION pipeline of our RAG system. It connects the first four stages:

    PDF -> text extraction -> chunking -> embeddings -> ChromaDB

After a PDF has been ingested, its chunks are stored in the vector
database and can be searched by retriever.py.
"""

import os
import sys

# Works both when imported from the project root (src.ingest)
# and when this file is run directly (python src/ingest.py)
try:
    from src.pdf_loader import extract_text_from_pdf
    from src.chunker import chunk_documents
    from src.embeddings import generate_embeddings
    from src.vector_store import VectorStore
except ImportError:
    from pdf_loader import extract_text_from_pdf
    from chunker import chunk_documents
    from embeddings import generate_embeddings
    from vector_store import VectorStore


def ingest_pdf(pdf_path, store=None):
    """
    Read a PDF and store its chunks and embeddings in the vector database.

    Args:
        pdf_path (str): Path to the PDF file.
        store (VectorStore, optional): The vector store to save into.
            If not supplied, the default VectorStore() is used.

    Returns:
        int: Number of chunks added (0 if the PDF has no usable text).

    Raises:
        FileNotFoundError, ValueError, RuntimeError: Passed on from the
            other modules (missing file, bad PDF, failed embedding request).
    """
    # 1. Use the default vector store if none was given
    if store is None:
        store = VectorStore()

    # 2. Extract the text of each page (pages without text are skipped)
    pages = extract_text_from_pdf(pdf_path)

    # Empty or image-only PDF: nothing to store, so stop here
    # (this also avoids a pointless API call)
    if not pages:
        return 0

    # 3. Split each page into smaller overlapping chunks
    chunks = chunk_documents(pages)

    if not chunks:
        return 0

    # 4. Convert every chunk's text into an embedding vector
    texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(texts)

    # 5. Save chunks + embeddings in ChromaDB
    return store.add_documents(chunks, embeddings)


# Command-line test: run from the project root with
#   python src/ingest.py error.pdf
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/ingest.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        # These two steps are local (no API calls); we run them here only
        # to show how many pages and chunks the PDF produces.
        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_documents(pages)

        # The real pipeline: extract -> chunk -> embed -> store
        stored = ingest_pdf(pdf_path)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        sys.exit(1)

    print("PDF filename:", os.path.basename(pdf_path))
    print("Pages extracted:", len(pages))
    print("Chunks created:", len(chunks))
    print("Chunks stored:", stored)

    if stored == 0:
        print("No text found in this PDF (it may be empty or scanned).")