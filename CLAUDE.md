# regulations-rag

RAG chatbot to query EU and national regulations, including Q&A, with source citations

## What the user does
The user opens a Streamlit chat, asks a question in natural language about the
ingested regulations, and gets back an answer grounded in the source documents,
with a collapsable citation for each source chunk used (regulation title, page,
chunk text).

## Stack
- Docling
- OpenAI text-embedding-3-small
- Qdrant (vector db, local via Docker)
- gpt-5-nano for generation
- Streamlit for the UI

## Important rules
- The LLM must answer ONLY using the retrieved chunks (never from its own knowledge)
- If the context is insufficient, it must say so explicitly, not make things up
- Every answer must report the citations it used: regulation title, page, chunk text
- Ingestion must check whether a document is already in Qdrant before re-embedding it

## Structure
data/raw/              source PDFs to ingest
src/ingestion.py       business logic: hash check, parse, chunk, embed, upsert in qdrant
src/retrieval.py       business logic: semantic search + generation
src/vector_store.py    wrapper: Qdrant connection and operations
                       (named vector_store.py, not qdrant_client.py, to avoid
                       shadowing the qdrant-client pip package — Streamlit
                       inserts src/ at the front of sys.path when running
                       src/app.py, which would otherwise cause a self-import)
src/app.py             Streamlit entrypoint (multipage nav)
src/pages/chat.py      UI: chat, calls retrieval.py, shows answer + citations
src/pages/ingest.py    UI: ingestion, calls ingestion.py, shows cost + Confirm button
evals/           evals and golden dataset
doc/            specs

## Future improvements (not now)
- Golden dataset + evals to compare gpt-5-nano vs a stronger model, and
  text-embedding-3-small vs -3-large
- Let the user upload new documents from the UI
- Let the user filter by regulation title (metadata filter)
- Let the user see which documents are currently loaded in Qdrant
- Cost dashboard