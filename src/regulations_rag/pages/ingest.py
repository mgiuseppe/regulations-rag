import uuid

import streamlit as st
from openai import OpenAI

from regulations_rag import ingestion
from regulations_rag.config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS
from regulations_rag.repositories.documents_repository import DocumentsRepository
from regulations_rag.repositories.vector_store import VectorStore

st.title("Ingest regulations")

@st.cache_data(show_spinner=False)
def _parse_and_chunk_cached(path_str: str, file_hash: str, model_name: str):
    from pathlib import Path
    return ingestion.parse_and_chunk(Path(path_str), EMBEDDING_MODELS[model_name])

if "uploaded_paths" not in st.session_state:
    st.session_state.uploaded_paths = []

# upload files on server
files_to_upload = st.file_uploader(
    "Upload PDF regulations", type="pdf", accept_multiple_files=True
)
if st.button("Upload", disabled=not files_to_upload):
    with st.spinner(f"Uploading..."):
        for uploaded in files_to_upload:
            ingestion.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            path = ingestion.RAW_DATA_DIR / uploaded.name
            path.write_bytes(uploaded.getvalue())
            if path not in st.session_state.uploaded_paths:
                st.session_state.uploaded_paths.append(path)
        st.rerun()

paths = st.session_state.uploaded_paths

if not paths:
    st.info("Upload a PDF above to get started.")
    st.stop()

if "ingest_titles" not in st.session_state:
    st.session_state.ingest_titles = {}

embedding_config = EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]
documents_repo = DocumentsRepository()
vector_store = VectorStore()
vector_store.ensure_collection()

to_process = []
total_tokens = 0
total_cost = 0.0

# Select files to append
for path in paths:
    file_hash = ingestion.compute_file_hash(path)

    st.subheader(path.name)

    default_title = st.session_state.ingest_titles.get(
        path.name, ingestion.candidate_title_from_filename(path.name)
    )
    title = st.text_input("Title", value=default_title, key=f"title_{path.name}")
    st.session_state.ingest_titles[path.name] = title

    match_result = ingestion.find_matches(file_hash, path.name, title, documents_repo)

    if match_result.status == "exact_match":
        exact_match = match_result.exact_matches[0]
        st.success(
            f"Already up to date — matches existing document "
            f"'{exact_match.title}' ({exact_match.source_file})"
        )
        continue

    action = "new"
    replace_target = None
    replace_point_ids = None

    if match_result.status == "ambiguous":
        ambiguous_matches = match_result.ambiguous_matches
        st.warning("This file looks like it might already be in the library:")
        for match in ambiguous_matches:
            st.write(
                f"- **{match.title}** ({match.source_file}) — "
                f"matched on: {', '.join(match.matched_on)}"
            )
        options = ["Undo (skip this file)", "Add as a new file"] + [
            f"Replace the file: {match.title} ({match.source_file})" for match in ambiguous_matches
        ]
        choice = st.radio(
            "What do you want to do?", options, key=f"action_{path.name}"
        )
        if choice == "Undo (skip this file)":
            continue
        if choice == "Add as a new file":
            action = "new"
        else:
            replace_target = ambiguous_matches[options.index(choice) - 2]
            action = "replace"
            # Snapshot which points to remove now — inserting the new points
            # first and deleting these specific IDs afterward.
            replace_point_ids = vector_store.get_document_point_ids(replace_target.document_id)
    else:
        st.write("New file — will be ingested")

    with st.spinner(f"Parsing {path.name}..."):
        chunks = _parse_and_chunk_cached(str(path), file_hash, embedding_config.name)

    tokens, cost = ingestion.estimate_cost(chunks, embedding_config)
    total_tokens += tokens
    total_cost += cost
    st.write(f"{tokens:,} tokens (~${cost:.4f})")

    replace_document_id = replace_target.document_id if replace_target else None

    to_process.append(
        {
            "path": path,
            "title": title,
            "chunks": chunks,
            "file_hash": file_hash,
            "action": action,
            "replace_point_ids": replace_point_ids,
            "replace_document_id": replace_document_id,
        }
    )

# Process files
if to_process:
    st.divider()
    st.write(f"**Total: {total_tokens:,} tokens (~${total_cost:.4f})**")

    if st.button("Confirm"):
        openai_client = OpenAI()
        progress = st.progress(0.0)
        errors = []
        for i, item in enumerate(to_process):
            with st.spinner(f"Embedding {item['path'].name}..."):
                try:
                    if item["action"] == "replace":
                        document_id = item["replace_document_id"]
                        ingestion.replace_document(
                            path=item["path"],
                            title=item["title"],
                            chunks=item["chunks"],
                            file_hash=item["file_hash"],
                            document_id=document_id,
                            embedding_config=embedding_config,
                            vector_store=vector_store,
                            openai_client=openai_client,
                            documents_repo=documents_repo,
                        )
                        # New points are inserted before old ones are deleted:
                        # Qdrant has no transactions, so this avoids a window
                        # where the document is missing entirely if something
                        # fails mid-way.
                        vector_store.delete_points(item["replace_point_ids"])
                    else:
                        document_id = str(uuid.uuid4())
                        ingestion.ingest_new_document(
                            path=item["path"],
                            title=item["title"],
                            chunks=item["chunks"],
                            file_hash=item["file_hash"],
                            document_id=document_id,
                            embedding_config=embedding_config,
                            vector_store=vector_store,
                            openai_client=openai_client,
                            documents_repo=documents_repo,
                        )
                except Exception as exc:
                    errors.append(f"{item['path'].name}: {exc}")
            progress.progress((i + 1) / len(to_process))
        st.session_state.uploaded_paths = []
        if errors:
            st.error("Some files failed to ingest:\n" + "\n".join(f"- {e}" for e in errors))
        else:
            st.success("Ingestion complete")
        st.rerun()
else:
    st.info("Nothing to ingest — all files are up to date.")
