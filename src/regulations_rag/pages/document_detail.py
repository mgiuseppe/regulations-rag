import streamlit as st

from regulations_rag import vector_store

document_id = st.query_params.get("document_id")

if not document_id:
    st.error("No document selected.")
    st.page_link("pages/documents.py", label="Back to documents", icon=":material/arrow_back:")
    st.stop()

qdrant_client = vector_store.get_client()
vector_store.ensure_collection(qdrant_client)

document = vector_store.get_document(qdrant_client, document_id)

if document is None:
    st.error("Document not found.")
    st.page_link("pages/documents.py", label="Back to documents", icon=":material/arrow_back:")
    st.stop()

st.page_link("pages/documents.py", label="Back to documents", icon=":material/arrow_back:")
st.title(document["title"])
st.write(f"**Source file:** {document['source_file']}")
st.caption(f"Document ID: {document['document_id']}")

if st.session_state.get("chunks_document_id") != document_id:
    st.session_state.chunks_document_id = document_id
    st.session_state.chunks = []
    st.session_state.chunks_cursor = None
    st.session_state.chunks_exhausted = False

if not st.session_state.chunks and not st.session_state.chunks_exhausted:
    points, next_cursor = vector_store.scroll_document_chunks(qdrant_client, document_id)
    st.session_state.chunks.extend(points)
    st.session_state.chunks_cursor = next_cursor
    st.session_state.chunks_exhausted = next_cursor is None

st.subheader(f"Chunks ({len(st.session_state.chunks)})")

for point in st.session_state.chunks:
    payload = point.payload or {}
    with st.expander(f"Chunk {payload.get('chunk_number')} — page {payload.get('page')}"):
        st.caption(f"Chunk ID: {point.id}")
        st.write(payload.get("text"))

if not st.session_state.chunks_exhausted:
    if st.button("Load more"):
        points, next_cursor = vector_store.scroll_document_chunks(
            qdrant_client, document_id, cursor=st.session_state.chunks_cursor
        )
        st.session_state.chunks.extend(points)
        st.session_state.chunks_cursor = next_cursor
        st.session_state.chunks_exhausted = next_cursor is None
        st.rerun()
else:
    st.caption("All chunks loaded.")
