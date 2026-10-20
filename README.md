# regulations-rag

RAG chatbot to query EU and national regulations, including Q&A, with source citations.

The user opens a Streamlit chat, asks a question in natural language about the
ingested regulations, and gets back an answer grounded in the source documents,
with a collapsible citation for each source chunk used (regulation title, page,
chunk text).

## Features

- Ingests PDF regulations (Docling) into a local Qdrant vector store
- Semantic search + generation grounded strictly in retrieved chunks
- Every answer includes citations (regulation title, page, chunk text)
- Skips re-embedding documents already present in Qdrant
- Tracks the document catalog (id, filename, title, hash, chunk count, status)
  in Postgres, browsable and sortable on the **Documents** page

## Tech stack

- [Docling](https://github.com/docling-project/docling) — PDF parsing
- OpenAI `text-embedding-3-small` — embeddings
- [Qdrant](https://qdrant.tech/) — vector database (local via Docker)
- [Postgres](https://www.postgresql.org/) via SQLAlchemy + Alembic — document catalog (local via Docker)
- OpenAI `gpt-5-nano` — answer generation
- [Streamlit](https://streamlit.io/) — UI

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- Docker (to run Qdrant and Postgres locally)
- An OpenAI API key

## Installation

```bash
# clone the repo
git clone <repo-url>
cd regulations-rag

# install dependencies (including dev dependencies)
uv sync
```

## Configuration

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-...
QDRANT_URL=http://localhost:6333
DATABASE_URL=postgresql+psycopg://regulations_rag:regulations_rag@localhost:5432/regulations_rag
```

## Running Qdrant and Postgres

```bash
docker compose up -d
```

This starts Qdrant (ports 6333/6334) and Postgres (port 5432) with the
credentials expected by the `DATABASE_URL` above, each with a persistent
Docker volume.

## Database schema

Schema changes are managed with Alembic — the app never calls `create_all()`.
After Postgres is up, apply migrations:

```bash
uv run alembic upgrade head
```

When you change the `DocumentModel` in `repositories/documents_repository.py`, generate a
new migration with:

```bash
uv run alembic revision --autogenerate -m "describe the change"
```

## Usage

```bash
# add source PDFs to data/raw/, then start the app
uv run streamlit run src/regulations_rag/app.py
```

Open the app in your browser, use the **Ingest** page to load documents into
Qdrant and Postgres, the **Documents** page to browse the catalog, and the
**Chat** page to ask questions.

## Project structure

```
data/raw/                                                source PDFs to ingest
alembic/                                                 Postgres schema migrations
src/regulations_rag/ingestion.py                         business logic: hash check, parse, chunk, embed, upsert in qdrant/postgres
src/regulations_rag/retrieval.py                         business logic: semantic search + generation
src/regulations_rag/repositories/vector_store.py         wrapper: Qdrant connection and operations
src/regulations_rag/repositories/documents_repository.py wrapper: Postgres connection and document catalog (only file that knows SQLAlchemy)
src/regulations_rag/app.py                               Streamlit entrypoint (multipage nav)
src/regulations_rag/pages/chat.py                        UI: chat, calls retrieval.py, shows answer + citations
src/regulations_rag/pages/ingest.py                      UI: ingestion, calls ingestion.py, shows cost + Confirm button
src/regulations_rag/pages/documents.py                   UI: paginated, sortable document catalog, calls documents_repository.py
evals/                                                   evals and golden dataset
doc/                                                      specs
tests/                                                    unit tests
```

## Development

```bash
# run tests
uv run pytest
```

## License

TBD

## Next Steps
The current pipeline retrieves small chunks that sometimes lack enough surrounding context, and can mix passages from different regulations.

The improvements below are planned in priority order.

Notes: to understand if we've to change the LLM or the semantic search we need the evaluation harness

### 1. Evaluation harness
- Define golden dataset stored as jsonl (15-20 questions, paired with article/document)
- Measure retrieval accuracy (recall@k)
- Setup evals with promptfoo
- Later: score with an LLM-as-a-judge (RAGAS)

### 2. Try different models (LLM or embedding)
- I'm currently using gpt-5-nano, we should try another one (only if retrieved chunks are the right ones)
- I'm currently using text-embedding-3-small try a larger one (only if chunks are missing / wrong)
  
### 3. Test small-to-big retrieval
decouple what we search on from what we feed the model.
- chunk-window (i-2...i+2)
- parent-document (retrieved from postgres by id) 

### 4. Prompt constraints
- Instruct the model to keep sources separate and cite which regulation each statement comes from

### 5. Hybrid search
- dense + sparse/BM25 (recommendended for legal text, because they depend on exact tokens that pure semantic search blurs, qdrant support sparse vector and you can fuse results with a technique called Reciprocal Rank Fusion)

### 6. Reranking
- re-ranking with cross-encoder
