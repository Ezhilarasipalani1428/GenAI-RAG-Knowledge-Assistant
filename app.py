"""
app.py

Streamlit web interface for the "Generative AI Knowledge Assistant with RAG".

Run from the project root with:
    streamlit run app.py

How the app uses the RAG pipeline:
  1. "Process PDF"  -> ingest_pdf(): extract text, chunk it, create
                       embeddings and store them in ChromaDB.
  2. "Ask"          -> answer_question(): retrieve the most relevant chunks,
                       send them to Gemini as context and return a grounded
                       answer with source references.
  3. "Clear"        -> VectorStore.clear(): remove everything from ChromaDB.

The Gemini API key stays in the .env file and is used only inside the
backend modules. It is never displayed in this app.
"""

import os

import streamlit as st

from src.ingest import ingest_pdf
from src.rag_pipeline import answer_question
from src.vector_store import VectorStore

# Folder used to temporarily save uploaded PDFs
DATA_FOLDER = "data"

# -----------------------------------------------------------------------------
# Page setup (must be the first Streamlit command)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Generative AI Knowledge Assistant", page_icon="📚")


@st.cache_resource
def get_store():
    """
    Create the VectorStore only once and reuse it.
    Streamlit re-runs this whole script on every click, and cache_resource
    stops us from reopening the database each time.
    """
    return VectorStore()


store = get_store()

# -----------------------------------------------------------------------------
# Title and description
# -----------------------------------------------------------------------------
st.title("📚 Generative AI Knowledge Assistant")
st.write(
    "Upload PDF documents and ask questions about them. This app uses "
    "**Retrieval-Augmented Generation (RAG)**: it finds the most relevant "
    "parts of your documents and uses them to generate an answer, showing "
    "the source file and page for every answer."
)

# A placeholder for the chunk count. We create it here (so it appears at the
# top of the page) but fill it at the very end of the script, after any
# upload / clear action has finished, so the number is always up to date.
chunk_count_box = st.empty()

st.divider()

# -----------------------------------------------------------------------------
# Section 1: Upload and process a PDF
# -----------------------------------------------------------------------------
st.header("1. Upload a PDF")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if st.button("Process PDF"):
    if uploaded_file is None:
        st.warning("Please choose a PDF file first.")
    else:
        # os.path.basename keeps only the file name (no folders), which is
        # safer. The name is also stored as the "source" of every chunk.
        file_name = os.path.basename(uploaded_file.name)
        temp_path = os.path.join(DATA_FOLDER, file_name)

        try:
            # Make sure the data folder exists, then save the upload to disk,
            # because ingest_pdf() needs a file path.
            os.makedirs(DATA_FOLDER, exist_ok=True)
            with open(temp_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getbuffer())

            # Run the ingestion pipeline:
            # PDF -> text -> chunks -> embeddings -> ChromaDB
            with st.spinner("Processing PDF... this may take a moment."):
                chunks_stored = ingest_pdf(temp_path, store=store)

            if chunks_stored == 0:
                st.warning(
                    "No readable text was found in this PDF. It may be "
                    "empty or a scanned document made of images."
                )
            else:
                st.success(
                    f"'{file_name}' processed successfully. "
                    f"{chunks_stored} chunks stored."
                )

        except (FileNotFoundError, ValueError, RuntimeError) as error:
            # Clear error messages raised by our own modules
            st.error(f"Could not process the PDF: {error}")
        except Exception:
            # Anything unexpected: show a friendly message, not a traceback
            st.error("Something went wrong while processing the PDF. Please try again.")
        finally:
            # The text is already stored in ChromaDB, so the temporary
            # PDF copy is no longer needed.
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

st.divider()

# -----------------------------------------------------------------------------
# Section 2: Ask a question
# -----------------------------------------------------------------------------
st.header("2. Ask a question")

question = st.text_area(
    "Your question",
    placeholder="e.g. What is this document about?",
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please type a question first.")
    elif store.count() == 0:
        st.warning("The knowledge base is empty. Please upload and process a PDF first.")
    else:
        try:
            # Run the RAG pipeline:
            # question -> retrieve chunks -> Gemini -> answer + sources
            with st.spinner("Searching the documents and generating an answer..."):
                result = answer_question(question, store=store)

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Sources")
            if result["sources"]:
                for source in result["sources"]:
                    st.write(f"- 📄 **{source['source']}**, page {source['page']}")
            else:
                st.info("No sources were found for this question.")

        except (ValueError, RuntimeError) as error:
            # Includes a missing API key, retrieval errors and Gemini errors
            st.error(f"Could not answer the question: {error}")
        except Exception:
            st.error("Something went wrong while answering. Please try again.")

st.divider()

# -----------------------------------------------------------------------------
# Section 3: Clear the knowledge base
# -----------------------------------------------------------------------------
st.header("3. Manage knowledge base")

if st.button("Clear Knowledge Base"):
    try:
        store.clear()
        st.success("Knowledge base cleared. All stored chunks were removed.")
    except Exception:
        st.error("Could not clear the knowledge base. Please try again.")

# -----------------------------------------------------------------------------
# Fill the chunk-count placeholder created at the top of the page
# -----------------------------------------------------------------------------
try:
    chunk_count_box.metric("Chunks currently stored", store.count())
except Exception:
    chunk_count_box.warning("Could not read the number of stored chunks.")