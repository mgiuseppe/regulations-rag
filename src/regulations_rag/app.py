import streamlit as st

ingest_page = st.Page("pages/ingest.py", title="Ingest", icon="📥")
chat_page = st.Page("pages/chat.py", title="Chat", icon="💬", default=True)

pg = st.navigation([chat_page, ingest_page])
pg.run()
