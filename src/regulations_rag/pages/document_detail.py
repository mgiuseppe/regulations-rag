import streamlit as st

from regulations_rag.repositories.documents_repository import DocumentsRepository
from regulations_rag.repositories.vector_store import VectorStore

document_id = st.query_params.get("document_id")

if not document_id:
    st.error("No document selected.")
    st.page_link("pages/documents.py", label="Back to documents", icon=":material/arrow_back:")
    st.stop()

documents_repo = DocumentsRepository()
document = documents_repo.get_by_id(document_id)

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

vector_store = VectorStore()
vector_store.ensure_collection()

with st.spinner("Loading chunks..."):
    chunks = vector_store.get_document_chunks(document_id)

st.subheader(f"Chunks ({len(chunks)})")

for point in chunks:
    payload = point.payload or {}
    with st.expander(f"Chunk {payload.get('chunk_number')} — page {payload.get('page')}"):
        st.caption(f"Chunk ID: {point.id}")
        st.write(payload.get("text"))
