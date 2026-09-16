import streamlit as st
import logging
import time

from regulations_rag.repositories.documents_repository import DocumentsRepository
from regulations_rag.repositories.vector_store import VectorStore

logger = logging.getLogger(__name__)

page_start = time.perf_counter()
logger.info("document_detail page: start")

document_id = st.query_params.get("document_id")

if not document_id:
    st.error("No document selected.")
    st.page_link("pages/documents.py", label="Back to documents", icon=":material/arrow_back:")
    st.stop()

repo_init = time.perf_counter()
documents_repo = DocumentsRepository()
logger.info("document_detail page: DocumentsRepository init took %.3fs", time.perf_counter() - repo_init)
repo_start = time.perf_counter()
document = documents_repo.get_by_id(document_id)
logger.info("document_detail page: get_by_id took %.3fs", time.perf_counter() - repo_start)

if document is None:
    st.error("Document not found.")
    st.page_link("pages/documents.py", label="Back to documents", icon=":material/arrow_back:")
    st.stop()

st.page_link("pages/documents.py", label="Back to documents", icon=":material/arrow_back:")
st.title(document.title)
st.write(f"**File name:** {document.filename}")
st.caption(f"Document ID: {document.id}")
st.caption(f"Hash: {document.content_hash}")
st.write(f"**Status:** {document.status}")
if document.error_message:
    st.error(document.error_message)

with st.spinner("Loading chunks..."):
    vector_init = time.perf_counter()
    vector_store = VectorStore()
    vector_store.ensure_collection()
    logger.info("document_detail page: VectorStore init took %.3fs", time.perf_counter() - vector_init)
    vector_start = time.perf_counter()
    chunks = vector_store.get_document_chunks(document_id)
    logger.info("document_detail page: get_document_chunks took %.3fs", time.perf_counter() - vector_start)


render_start = time.perf_counter()

st.subheader(f"Chunks ({len(chunks)})")

for chunk in chunks:
    with st.expander(f"Chunk {chunk.chunk_number} — page {chunk.page}"):
        st.caption(f"Chunk ID: {chunk.id}")
        st.write(chunk.text)

logger.info(
    "document_detail page: render took %.3fs, total %.3fs",
    time.perf_counter() - render_start,
    time.perf_counter() - page_start,
)
