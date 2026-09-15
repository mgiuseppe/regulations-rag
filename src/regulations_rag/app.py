import streamlit as st

ingest_page = st.Page("pages/ingest.py", title="Ingest", icon="📥")
documents_page = st.Page("pages/documents.py", title="Documents", icon="📄")
chat_page = st.Page("pages/chat.py", title="Chat", icon="💬", default=True)

pg = st.navigation([chat_page, ingest_page, documents_page])
pg.run()
