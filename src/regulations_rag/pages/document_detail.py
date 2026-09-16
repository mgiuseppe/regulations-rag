import streamlit as st

from regulations_rag import vector_store
from regulations_rag.documents_repository import DocumentsRepository

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

qdrant_client = vector_store.get_client()
vector_store.ensure_collection(qdrant_client)

with st.spinner("Loading chunks..."):
    chunks = vector_store.get_document_chunks(qdrant_client, document_id)

st.subheader(f"Chunks ({len(chunks)})")

for point in chunks:
    payload = point.payload or {}
    with st.expander(f"Chunk {payload.get('chunk_number')} — page {payload.get('page')}"):
        st.caption(f"Chunk ID: {point.id}")
        st.write(payload.get("text"))
