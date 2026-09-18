# regulations-rag

RAG chatbot to query EU and national regulations, including Q&A, with source citations

## What the user does
The user opens a Streamlit chat, asks a question in natural language about the ingested regulations, and gets back an answer grounded in the source documents, with a collapsable citation for each source chunk used (regulation title, page, chunk text).

## Stack
- Docling
- OpenAI text-embedding-3-small
- Qdrant (vector db, local via Docker)
- Postgres via SQLAlchemy + Alembic (document catalog, local via Docker)
- gpt-5-nano for generation
- Streamlit for the UI

## Workflow
- Never commit directly to `main`. Every change is done on a feature branch
  created from `main` (e.g. `feature/short-description`, `fix/short-description`).
- Open a pull request into `main` and merge through the PR (not a local merge/push
  to `main`), even for small changes.
- For any multi-part or ambiguous change, present a file-by-file plan (and call
  out open questions, e.g. tie-breaking rules) before writing code, and wait for
  explicit go-ahead.
- Match complexity to the actual data volume — don't add pagination, lazy-loading,
  or other speculative machinery for collections that are small in practice (e.g.
  a document's chunk list). Remove it if it turns out not to be solving a real
  problem.
- Run the full test suite after every change; when functionality is deleted,
  delete its tests in the same change rather than leaving them stale.
- Smoke-test changes in the running app (not just `pytest`), especially anything
  touching retrieval grounding or citation rendering — automated tests don't
  catch a wrong answer that's merely well-typed.
- Delete a feature's local branch, remote branch, and worktree only after
  confirming its PR is actually merged.

## Important rules
- The LLM must answer ONLY using the retrieved chunks (never from its own knowledge)
- If the context is insufficient, it must say so explicitly, not make things up
- Every answer must report the citations it used: regulation title, page, chunk text
- Ingestion must check whether a document is already in Qdrant before re-embedding it
- `repositories/documents_repository.py` is the only module that knows about SQLAlchemy —
  pages and ingestion logic exchange plain `DocumentRecord` objects with it
- Ingestion order for a new document: insert the Postgres row as `pending`
  before embedding, write chunks to Qdrant, then update the row to `indexed`
  with the final chunk count; on any failure set it to `failed` with the
  error message. A row is never left `pending` without a reason.
- Updating an existing document keeps the same `document_id`: write the new
  chunks to Qdrant first, update the Postgres row, and only then delete the
  old Qdrant chunks (Qdrant has no transactions)
- Schema changes go through Alembic migrations, never `create_all()`
- The Postgres connection string comes from the `DATABASE_URL` environment
  variable — no localhost defaults in application code
- Every external SDK (OpenAI, Qdrant, SQLAlchemy) is owned by exactly one
  module — pages and business logic never import the SDK directly, only
  through that module's wrapper. `documents_repository.py` already does this
  for SQLAlchemy and `vector_store.py` for `qdrant-client`
- Wrapper/client modules are named by role, not vendor (e.g. a shared OpenAI
  wrapper should be `llm_client`, not `openai_client`), so the app can swap
  providers later without a rename.
- Stateful wrapper clients follow one pattern: a `@cache`-decorated
  module-level factory (e.g. `_get_vector_store()`) plus a class
  `__init__(self, resource=None)` that defaults to the cached singleton —
  this is what lets tests inject a fake/mock without touching the real
  client.
- A service's public surface is a single orchestrating method (e.g.
  `RetrievalService.answer()`); its steps are private (`_`-prefixed)
  methods — still directly unit-testable in Python, but not part of the
  page-facing API.
- Pages are pure UI: they render, take input, and call one service/module
  method — RAG orchestration (embed → search → build prompt → generate)
  lives in `services/retrieval.py`, never in `pages/chat.py`.
- Document metadata (title, filename, status, chunk count, content hash)
  always comes from `documents_repository`/Postgres; only chunk text/page
  content comes from Qdrant. Never source catalog metadata from a Qdrant
  payload.
- Document identity is never the filename alone (users rename files).
  Matching an upload against existing documents checks SHA hash, filename,
  and title: an exact hash match needs no prompt (already up to date); any
  other partial match must ask the user to choose (undo / add as new /
  replace).
- Every ingested document carries a UUID4 `document_id` in its Qdrant
  payload as its stable key (legacy documents without one fall back to
  `source_file`). Replacing a document always mints a fresh `document_id` —
  it never reuses the old one.
- When replacing a document, delete the old Qdrant points by their exact
  point IDs captured *before* the new insert — never by a field filter
  (e.g. `source_file`), since the new points can share that field value and
  get deleted too.
- OpenAI embedding calls must stay under the API's 300k-tokens and
  2048-items per-request limits (keep a safety margin, e.g. ~250k tokens) —
  split `embed_chunks` into multiple batched calls and concatenate the
  vectors in order.
- Qdrant's `order_by` is incompatible with cursor-based `scroll` pagination
  (`next_page_offset` comes back `null` once `order_by` is set). For
  ordered results, fetch everything via the ID-based cursor and sort
  client-side in Python instead.
- Never name a local wrapper module the same as the pip package it wraps —
  Streamlit puts the entrypoint script's own directory at the front of
  `sys.path`, so a same-named file shadows the real package and self-imports
  forever (why the Qdrant wrapper is `vector_store.py`, not
  `qdrant_client.py`).

## Structure
`src/regulations_rag/` is an installable package (see `pyproject.toml`'s
`[build-system]`/`[tool.hatch.build.targets.wheel]`), installed editable by
`uv sync` — this is what lets every module import it as `regulations_rag`
regardless of how or from where a script is run, with no `sys.path` hacks.

- data/raw/: source PDFs to ingest
- alembic/: Postgres schema migrations (autogenerated from documents_repository.Base)
- src/regulations_rag/ingestion.py: business logic - hash check, parse, chunk, embed, upsert in qdrant/postgres
- src/regulations_rag/services/retrieval.py: business logic - semantic search + generation, as RetrievalService (owns the OpenAI and VectorStore clients, both cached module-level singletons. Pages never construct or import those clients directly)
- src/regulations_rag/repositories/vector_store.py: wrapper - Qdrant connection and operations
- src/regulations_rag/repositories/documents_repository.py: wrapper - Postgres connection + document catalog repository (the only module that knows about SQLAlchemy)
- src/regulations_rag/app.py: Streamlit entrypoint (multipage nav)
- src/regulations_rag/pages/chat.py: UI - chat, calls RetrievalService, shows answer + citations
- src/regulations_rag/pages/ingest.py: UI - upload + ingestion, calls ingestion.py, shows cost + Confirm button (only processes files uploaded in the current browser session, never scans data/raw/ automatically)
- src/regulations_rag/pages/documents.py: UI - sortable document catalog, calls documents_repository.py
- src/regulations_rag/pages/document_detail.py: UI - one document's metadata (from documents_repository) + its chunks in page order (from vector_store)
- evals/: evals and golden dataset
- doc/: specs

## Streamlit-specific gotchas
- Cache expensive parsing (Docling) with `st.cache_data` — otherwise an
  unrelated widget interaction (e.g. typing in a title field) triggers a
  full rerun and re-parses the PDF from scratch.
- Show `st.spinner` around any retrieval/fetch that isn't instant.
- A file uploader needs an explicit submit/"Upload" button, disabled until
  at least one file is selected — it must not auto-trigger ingestion on
  selection.
- If a function you just added to an already-imported module raises
  `AttributeError: module has no attribute ...`, that's usually Streamlit's
  stale module cache, not a real bug — a full app restart is needed,
  "Rerun" from the menu isn't enough.

## Future improvements (not now)
- Golden dataset + evals to compare gpt-5-nano vs a stronger model, and
  text-embedding-3-small vs -3-large
- Consolidate the OpenAI SDK behind a single `llm_client` wrapper module
  (currently imported directly in `ingestion.py`, `services/retrieval.py`,
  and `pages/ingest.py`)
- Let the user filter by regulation title (metadata filter)
- Cost dashboard