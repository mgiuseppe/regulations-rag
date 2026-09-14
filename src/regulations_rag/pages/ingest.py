import uuid

import streamlit as st
from openai import OpenAI

from regulations_rag import ingestion, vector_store
from regulations_rag.config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS

st.title("Ingest regulations")

embedding_config = EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]


@st.cache_data(show_spinner=False)
def _parse_and_chunk_cached(path_str: str, file_hash: str, model_name: str):
    from pathlib import Path
    return ingestion.parse_and_chunk(Path(path_str), EMBEDDING_MODELS[model_name])

qdrant_client = vector_store.get_client()
vector_store.ensure_collection(qdrant_client)

if "uploaded_paths" not in st.session_state:
    st.session_state.uploaded_paths = []

uploaded_files = st.file_uploader(
    "Upload PDF regulations", type="pdf", accept_multiple_files=True
)
if st.button("Upload", disabled=not uploaded_files):
    ingestion.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for uploaded in uploaded_files:
        path = ingestion.RAW_DATA_DIR / uploaded.name
        path.write_bytes(uploaded.getvalue())
        if path not in st.session_state.uploaded_paths:
            st.session_state.uploaded_paths.append(path)
    st.rerun()

files = st.session_state.uploaded_paths

if not files:
    st.info("Upload a PDF above to get started.")
    st.stop()

if "ingest_titles" not in st.session_state:
    st.session_state.ingest_titles = {}

to_process = []
total_tokens = 0
total_cost = 0.0

for path in files:
    file_hash = ingestion.compute_file_hash(path)

    st.subheader(path.name)

    default_title = st.session_state.ingest_titles.get(
        path.name, ingestion.candidate_title_from_filename(path.name)
    )
    title = st.text_input("Title", value=default_title, key=f"title_{path.name}")
    st.session_state.ingest_titles[path.name] = title

    matches = vector_store.find_matches(qdrant_client, file_hash, path.name, title)
    exact_match = next((m for m in matches if "hash" in m["matched_on"]), None)

    if exact_match:
        st.success(
            f"Already up to date — matches existing document "
            f"'{exact_match['title']}' ({exact_match['source_file']})"
        )
        continue

    action = "new"
    replace_target = None
    replace_point_ids = None

    if matches:
        st.warning("This file looks like it might already be in the library:")
        for match in matches:
            st.write(
                f"- **{match['title']}** ({match['source_file']}) — "
                f"matched on: {', '.join(match['matched_on'])}"
            )
        options = ["Undo (skip this file)", "Add as a new file"] + [
            f"Replace the file: {match['title']} ({match['source_file']})" for match in matches
        ]
        choice = st.radio(
            "What do you want to do?", options, key=f"action_{path.name}"
        )
        if choice == "Undo (skip this file)":
            continue
        if choice == "Add as a new file":
            action = "new"
        else:
            replace_target = matches[options.index(choice) - 2]
            action = "replace"
            # Snapshot which points to remove now — deleting by key_field/key
            # after the new points are inserted could also catch the new
            # points if they share the same source_file.
            replace_point_ids = vector_store.get_document_point_ids(
                qdrant_client, replace_target["key_field"], replace_target["key"]
            )
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
            "action": action,
            "replace_point_ids": replace_point_ids,
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
                # Every ingested document gets its own fresh GUID as document_id,
                # regardless of whether it's replacing an older document.
                document_id = str(uuid.uuid4())

                ingestion.ingest_file(
                    path=item["path"],
                    title=item["title"],
                    chunks=item["chunks"],
                    file_hash=item["file_hash"],
                    document_id=document_id,
                    embedding_config=embedding_config,
                    qdrant_client=qdrant_client,
                    openai_client=openai_client,
                )

                # Insert before delete: Qdrant has no transactions, so this
                # avoids a window where the document is missing entirely if
                # something fails mid-way.
                if item["action"] == "replace":
                    vector_store.delete_points(qdrant_client, item["replace_point_ids"])
            progress.progress((i + 1) / len(to_process))
        st.session_state.uploaded_paths = []
        st.success("Ingestion complete")
        st.rerun()
else:
    st.info("Nothing to ingest — all files are up to date.")
