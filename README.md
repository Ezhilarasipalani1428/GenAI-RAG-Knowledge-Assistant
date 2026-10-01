# Generative AI Knowledge Assistant with RAG

A Generative AI Knowledge Assistant that allows users to upload PDF documents and ask questions about their content.

The project uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from uploaded documents and provide grounded answers using Google's Gemini models. Answers also include the source file and page references used during retrieval.

## Project Overview

Traditional chatbots may generate answers using general knowledge, which can lead to incorrect or unsupported information.

This project follows a RAG approach:

**PDF → Text Extraction → Chunking → Embeddings → Vector Database → Retrieval → Gemini → Answer + Sources**

Instead of asking the language model to answer from general knowledge, the system first searches the uploaded documents for relevant content and then provides that content to Gemini as context.

## Features

* Upload PDF documents through a Streamlit interface
* Extract text from PDF files
* Split documents into smaller overlapping chunks
* Generate vector embeddings using Gemini
* Store document embeddings in ChromaDB
* Retrieve relevant document chunks for a question
* Generate answers using Gemini
* Display source filename and page number
* Local persistent vector database
* API key stored securely using environment variables

## How RAG Works

### 1. PDF Upload

The user uploads a PDF through the Streamlit application.

### 2. Text Extraction

PyMuPDF extracts readable text from each page.

Each page keeps metadata such as:

* Source filename
* Page number

### 3. Chunking

Large page text is divided into smaller overlapping chunks.

The current default configuration is:

* Chunk size: 1000 characters
* Chunk overlap: 200 characters

The overlap helps preserve context between neighboring chunks.

### 4. Embeddings

Each chunk is converted into a numerical vector using the Gemini embedding model.

The embedding represents the semantic meaning of the text.

### 5. Vector Storage

The embeddings and their corresponding text and metadata are stored in ChromaDB.

### 6. Retrieval

When the user asks a question, the question is also converted into an embedding.

ChromaDB searches for the most semantically similar document chunks.

### 7. Generation

The retrieved chunks are passed to Gemini as document context.

Gemini generates an answer using the retrieved context.

### 8. Source References

The application displays the source filename and page number associated with the retrieved content.

## Architecture / Workflow

```text
                    ┌─────────────────┐
                    │   PDF Upload    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Text Extraction │
                    │    PyMuPDF      │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │    Chunking     │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Gemini Embedding│
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │    ChromaDB     │
                    └────────┬────────┘
                             │
                             │
User Question ───────→ Embedding
                             │
                             ↓
                    ┌─────────────────┐
                    │    Retrieval    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Retrieved       │
                    │ Document Context│
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Gemini Generator│
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Answer + Sources│
                    └─────────────────┘
```

## Tech Stack

| Technology        | Purpose                              |
| ----------------- | ------------------------------------ |
| Python 3.12       | Application development              |
| Streamlit         | Web interface                        |
| PyMuPDF           | PDF text extraction                  |
| ChromaDB          | Vector storage and similarity search |
| Google GenAI SDK  | Gemini API integration               |
| Gemini Embeddings | Text embeddings                      |
| Gemini            | Answer generation                    |
| python-dotenv     | Environment variable management      |
| Git/GitHub        | Version control                      |

## Project Structure

```text
GenAI-RAG-Knowledge-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── data/
│   └── Local ChromaDB data
│
└── src/
    ├── pdf_loader.py
    ├── chunker.py
    ├── embeddings.py
    ├── vector_store.py
    ├── retriever.py
    ├── ingest.py
    └── rag_pipeline.py
```

### Main Files

**`app.py`**

Provides the Streamlit user interface for uploading PDFs, processing documents, asking questions, and managing the knowledge base.

**`src/pdf_loader.py`**

Extracts readable text from PDF pages and stores page-level metadata.

**`src/chunker.py`**

Splits extracted text into smaller overlapping chunks.

**`src/embeddings.py`**

Generates Gemini embeddings for document chunks and user questions.

**`src/vector_store.py`**

Stores embeddings and document information in ChromaDB and performs similarity searches.

**`src/retriever.py`**

Converts a user question into an embedding and retrieves the most relevant document chunks.

**`src/ingest.py`**

Connects PDF extraction, chunking, embedding generation, and vector storage into one ingestion pipeline.

**`src/rag_pipeline.py`**

Combines retrieval with Gemini generation to produce grounded answers and source references.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Ezhilarasipalani1428/GenAI-RAG-Knowledge-Assistant.git
cd GenAI-RAG-Knowledge-Assistant
```

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

## Environment Variable Configuration

Create a file named:

```text
.env
```

in the project root.

Add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

The repository includes `.env.example` as a safe configuration template.

**Never commit your real `.env` file or API key to GitHub.**

The `.gitignore` file is configured to prevent `.env` from being tracked.

## How to Run

From the project root, with the virtual environment activated:

```powershell
python -m streamlit run app.py
```

Streamlit will start the application and provide a local URL in the terminal.

## Example Usage

1. Start the Streamlit application.
2. Upload a PDF document.
3. Click **Process PDF**.
4. Wait for the document to be extracted, chunked, embedded, and stored.
5. Enter a question about the uploaded document.
6. Click **Ask**.
7. The application retrieves relevant document chunks.
8. Gemini generates an answer using the retrieved context.
9. The application displays the answer and source page references.

Example question:

```text
What is this document about?
```

## Limitations

* The current PDF pipeline extracts readable text but does not perform OCR on image-only scanned PDFs.
* Answer quality depends on the quality and relevance of retrieved chunks.
* Gemini API availability and usage limits can affect generation requests.
* The vector database is stored locally.
* The application currently focuses on document question answering rather than general-purpose conversation.
* Source references identify the retrieved filename and page but are not formal academic citations.

## Future Enhancements

Possible future improvements include:

* Support for multiple uploaded documents
* Better document management
* Improved chunking strategies
* Reranking retrieved results
* Conversation history
* Streaming answers
* More detailed citation display
* OCR support for scanned documents
* Authentication and multi-user support
* Document summarization
* Improved user interface
* Evaluation of retrieval and answer quality

These are planned enhancements and are not currently implemented.

## Security Note

The Gemini API key is stored in a local `.env` file and loaded through environment variables.

The `.env` file is excluded from Git using `.gitignore`.

Do not place API keys, passwords, tokens, or other secrets directly in source code or commit them to a public repository.

## License

This project is currently intended as a college learning and portfolio project.
