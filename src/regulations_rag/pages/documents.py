import math

import streamlit as st

from regulations_rag.documents_repository import DocumentsRepository

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

documents, total = documents_repo.list_documents(
    page=page, page_size=page_size, sort_by=sort_by, sort_order=sort_order
)

if total == 0:
    st.info("No documents ingested yet.")
    st.stop()

total_pages = max(1, math.ceil(total / page_size))
page = min(page, total_pages)
st.session_state.documents_page = page

st.write(f"{total} document(s) — page {page} of {total_pages}")

header = st.columns([3, 3, 2, 1, 1, 1])
for col, label in zip(header, ["Title", "File name", "Hash", "Chunks", "Status", ""]):
    col.write(f"**{label}**")

for doc in documents:
    row = st.columns([3, 3, 2, 1, 1, 1])
    row[0].write(doc.title)
    row[1].write(doc.filename)
    row[2].write(doc.content_hash[:12] + "…")
    row[3].write(doc.chunk_count)
    row[4].write(doc.status)
    row[5].page_link(
        "pages/document_detail.py",
        label="View",
        icon=":material/chevron_right:",
        query_params={"document_id": doc.id},
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
