import streamlit as st

ingest_page = st.Page("pages/ingest.py", title="Ingest", icon="📥")
chat_page = st.Page("pages/chat.py", title="Chat", icon="💬", default=True)
documents_page = st.Page("pages/documents.py", title="Documents", icon="📄")
document_detail_page = st.Page(
    "pages/document_detail.py",
    title="Document detail",
    icon="📄",
    url_path="document_detail",
    visibility="hidden",
)

pg = st.navigation([chat_page, documents_page, ingest_page, document_detail_page])
pg.run()
