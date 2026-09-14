import streamlit as st
from openai import OpenAI

from src import ingestion, vector_store
from src.config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS

st.title("Ingest regulations")

embedding_config = EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]


@st.cache_data(show_spinner=False)
def _parse_and_chunk_cached(path_str: str, file_hash: str, model_name: str):
    from pathlib import Path

    return ingestion.parse_and_chunk(Path(path_str), EMBEDDING_MODELS[model_name])

qdrant_client = vector_store.get_client()
vector_store.ensure_collection(qdrant_client)

files = ingestion.scan_raw_files()

if not files:
    st.info("No PDF files found in data/raw/.")
    st.stop()

if "ingest_titles" not in st.session_state:
    st.session_state.ingest_titles = {}

to_process = []
total_tokens = 0
total_cost = 0.0

for path in files:
    file_hash = ingestion.compute_file_hash(path)
    existing_document = vector_store.get_existing_document(qdrant_client, path.name)
    existing_hash = existing_document.get("content_hash") if existing_document else None
    status = ingestion.check_status(existing_hash, file_hash)

    if status == "up_to_date":
        st.subheader(f"{path.name} — {existing_document.get('title')}")
        st.success("Already up to date")
        continue

    st.subheader(path.name)

    default_title = st.session_state.ingest_titles.get(
        path.name, ingestion.candidate_title_from_filename(path.name)
    )
    title = st.text_input("Title", value=default_title, key=f"title_{path.name}")
    st.session_state.ingest_titles[path.name] = title

    if status == "stale":
        st.warning("File changed since last ingestion — will be re-ingested")
    else:
        st.write("New file — will be ingested")

    with st.spinner(f"Parsing {path.name}..."):
        chunks = _parse_and_chunk_cached(str(path), file_hash, embedding_config.name)

    tokens, cost = ingestion.estimate_cost(chunks, embedding_config)
    total_tokens += tokens
    total_cost += cost
    st.write(f"{tokens:,} tokens (~${cost:.4f})")

    to_process.append(
        {
            "path": path,
            "title": title,
            "chunks": chunks,
            "file_hash": file_hash,
            "status": status,
        }
    )

if to_process:
    st.divider()
    st.write(f"**Total: {total_tokens:,} tokens (~${total_cost:.4f})**")

    if st.button("Confirm"):
        openai_client = OpenAI()
        progress = st.progress(0.0)
        for i, item in enumerate(to_process):
            with st.spinner(f"Embedding {item['path'].name}..."):
                ingestion.ingest_file(
                    path=item["path"],
                    title=item["title"],
                    chunks=item["chunks"],
                    file_hash=item["file_hash"],
                    status=item["status"],
                    embedding_config=embedding_config,
                    qdrant_client=qdrant_client,
                    openai_client=openai_client,
                )
            progress.progress((i + 1) / len(to_process))
        st.success("Ingestion complete")
        st.rerun()
else:
    st.info("Nothing to ingest — all files are up to date.")
