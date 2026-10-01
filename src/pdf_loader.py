"""
pdf_loader.py

Reads a PDF file page by page and returns the text of each page
together with simple metadata (file name and page number).
This is the first step of our RAG pipeline.
"""

import os
import sys

import pymupdf  # PyMuPDF


def extract_text_from_pdf(pdf_path):
    """
    Extract text from every page of a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list[dict]: One dictionary per page that contains useful text:
            {
                "text": "text of the page",
                "metadata": {"source": "file.pdf", "page": 1}
            }

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF or is password protected.
        RuntimeError: If the PDF is corrupted or cannot be read.
    """
    # 1. Check that the file exists and is a PDF
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"File is not a PDF: {pdf_path}")

    # We store only the file name (not the full path) as the source
    file_name = os.path.basename(pdf_path)
    pages = []

    try:
        # 2. Open the PDF ("with" closes the file automatically)
        with pymupdf.open(pdf_path) as pdf:

            # Password-protected PDFs cannot be read without a password
            if pdf.needs_pass:
                raise ValueError(f"PDF is password protected: {file_name}")

            # 3. Go through the PDF page by page
            #    enumerate(..., start=1) makes page numbers start from 1
            for page_number, page in enumerate(pdf, start=1):

                # 4. Extract the text of the current page
                text = page.get_text("text").strip()

                # 5. Skip pages with no useful text
                #    (blank pages, or scanned pages that contain only images)
                if not text:
                    continue

                # 6. Save the text along with its metadata
                pages.append({
                    "text": text,
                    "metadata": {
                        "source": file_name,
                        "page": page_number,
                    },
                })

    except ValueError:
        # Our own errors (e.g. password protected): pass them on unchanged
        raise
    except Exception as error:
        # Any other PyMuPDF error, e.g. a corrupted PDF
        raise RuntimeError(f"Could not read '{file_name}': {error}") from error

    return pages


# Simple test: run this file directly from the project root
#   python src/pdf_loader.py path/to/your_file.pdf
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/pdf_loader.py <path_to_pdf>")
        sys.exit(1)

    try:
        result = extract_text_from_pdf(sys.argv[1])
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        sys.exit(1)

    print(f"Pages with text: {len(result)}")
    for item in result[:2]:  # show only the first 2 pages
        print("-" * 40)
        print("Metadata:", item["metadata"])
        print("Text preview:", item["text"][:200])