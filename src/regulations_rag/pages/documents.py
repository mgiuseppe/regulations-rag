import streamlit as st

from regulations_rag import vector_store

st.title("Documents")

qdrant_client = vector_store.get_client()
vector_store.ensure_collection(qdrant_client)

documents = vector_store.list_documents(qdrant_client)

if not documents:
    st.info("No documents ingested yet.")
    st.stop()

for document in documents:
    with st.container(border=True):
        cols = st.columns([4, 1])
        with cols[0]:
            st.write(f"**{document['title']}**")
            st.caption(document["source_file"])
        with cols[1]:
            if document["key_field"] == "document_id":
                st.page_link(
                    "pages/document_detail.py",
                    label="View details",
                    icon=":material/chevron_right:",
                    query_params={"document_id": document["key"]},
                )
            else:
                st.caption("Details unavailable for legacy documents")
