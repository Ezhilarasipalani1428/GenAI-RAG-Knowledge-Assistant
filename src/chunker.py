"""
chunker.py

Splits the page text returned by extract_text_from_pdf() into smaller,
overlapping chunks. Each chunk keeps the original source and page number
and gets its own chunk_id.
This is the second step of our RAG pipeline.
"""


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split the text of each page into smaller chunks (character-based).

    Args:
        documents (list[dict]): Output of extract_text_from_pdf(), e.g.
            [{"text": "...", "metadata": {"source": "a.pdf", "page": 1}}]
        chunk_size (int): Maximum number of characters in one chunk.
        chunk_overlap (int): Number of characters shared by two
            consecutive chunks.

    Returns:
        list[dict]: One dictionary per chunk:
            {
                "text": "chunk text",
                "metadata": {
                    "source": "a.pdf",
                    "page": 1,
                    "chunk_id": "a.pdf_p1_c1"
                }
            }

    Raises:
        ValueError: If chunk_size or chunk_overlap are invalid.
    """
    # 1. Validate the arguments
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # 2. How far we move forward for each new chunk.
    #    Example: size 1000, overlap 200 -> step 800, so every new chunk
    #    starts 800 characters after the previous one and the last 200
    #    characters of a chunk are repeated at the start of the next one.
    step = chunk_size - chunk_overlap

    chunks = []

    # 3. Process each page separately so chunks never mix two pages
    for document in documents:
        text = document["text"]
        metadata = document["metadata"]

        chunk_number = 0  # counts chunks inside the current page
        start = 0         # position where the current chunk begins

        while start < len(text):
            # 4. Cut out one chunk of at most chunk_size characters
            chunk_text = text[start:start + chunk_size].strip()

            # 5. Skip chunks that are empty after removing extra spaces
            if chunk_text:
                chunk_number += 1

                # Copy the original metadata so we do not change it,
                # then add a unique chunk_id
                chunk_metadata = dict(metadata)
                chunk_metadata["chunk_id"] = (
                    f"{metadata['source']}_p{metadata['page']}_c{chunk_number}"
                )

                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata,
                })

            # 6. Stop if this chunk already reached the end of the text
            #    (avoids a useless last chunk made only of overlap)
            if start + chunk_size >= len(text):
                break

            # 7. Move forward to the start of the next chunk
            start += step

    return chunks


# Simple test: run from the project root with
#   python src/chunker.py
if __name__ == "__main__":
    sample = [{
        "text": "A" * 250,
        "metadata": {"source": "demo.pdf", "page": 1},
    }]

    result = chunk_documents(sample, chunk_size=100, chunk_overlap=20)

    print(f"Total chunks: {len(result)}")
    for item in result:
        print(item["metadata"], "-> length:", len(item["text"]))