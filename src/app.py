import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import streamlit as st

ingest_page = st.Page("pages/ingest.py", title="Ingest", icon="📥")
chat_page = st.Page("pages/chat.py", title="Chat", icon="💬", default=True)

pg = st.navigation([chat_page, ingest_page])
pg.run()
