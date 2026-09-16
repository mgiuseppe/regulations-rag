import logging
import math
import time

import streamlit as st

from regulations_rag.documents_repository import DocumentsRepository

logger = logging.getLogger(__name__)

page_start = time.perf_counter()
logger.info("documents page: start")

st.title("Documents")

documents_repo = DocumentsRepository()

PAGE_SIZE_OPTIONS = [5, 10, 20, 50]
SORT_OPTIONS = {
    "Title": "title",
    "Creation date": "created_at",
    "File name": "file_name",
}

col1, col2, col3 = st.columns(3)
with col1:
    page_size = st.selectbox("Items per page", PAGE_SIZE_OPTIONS, index=0)
with col2:
    sort_label = st.selectbox("Sort by", list(SORT_OPTIONS.keys()), index=0)
    sort_by = SORT_OPTIONS[sort_label]
with col3:
    sort_order = st.selectbox("Order", ["Ascending", "Descending"], index=0)
    sort_order = "asc" if sort_order == "Ascending" else "desc"

sort_key = (page_size, sort_by, sort_order)
if st.session_state.get("documents_sort_key") != sort_key:
    st.session_state.documents_sort_key = sort_key
    st.session_state.documents_page = 1

page = st.session_state.get("documents_page", 1)

with st.spinner("Loading documents..."):
    repo_start = time.perf_counter()
    documents, total = documents_repo.list_documents(
        page=page, page_size=page_size, sort_by=sort_by, sort_order=sort_order
    )
    logger.info("documents page: list_documents took %.3fs", time.perf_counter() - repo_start)

if total == 0:
    st.info("No documents ingested yet.")
    st.stop()

total_pages = max(1, math.ceil(total / page_size))
page = min(page, total_pages)
st.session_state.documents_page = page

st.write(f"{total} document(s) — page {page} of {total_pages}")

header = st.columns([3, 3, 1, 1])
for col, label in zip(header, ["Title", "File name", "Chunks", "Status"]):
    col.write(f"**{label}**")

render_start = time.perf_counter()
for doc in documents:
    row = st.columns([3, 3, 1, 1])
    row[0].page_link(
        "pages/document_detail.py",
        label=doc.title,
        icon=":material/chevron_right:",
        query_params={"document_id": doc.id},
    )
    row[1].write(doc.filename)
    row[2].write(doc.chunk_count)
    row[3].write(doc.status)
logger.info(
    "documents page: render loop took %.3fs, total %.3fs",
    time.perf_counter() - render_start,
    time.perf_counter() - page_start,
)

nav_prev, nav_info, nav_next = st.columns([1, 2, 1])
with nav_prev:
    if st.button("Previous", disabled=page <= 1):
        st.session_state.documents_page = page - 1
        st.rerun()
with nav_next:
    if st.button("Next", disabled=page >= total_pages):
        st.session_state.documents_page = page + 1
        st.rerun()
