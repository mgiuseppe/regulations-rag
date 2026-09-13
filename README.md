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

## Tech stack

- [Docling](https://github.com/docling-project/docling) — PDF parsing
- OpenAI `text-embedding-3-small` — embeddings
- [Qdrant](https://qdrant.tech/) — vector database (local via Docker)
- OpenAI `gpt-5-nano` — answer generation
- [Streamlit](https://streamlit.io/) — UI

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- Docker (to run Qdrant locally)
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
```

## Running Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

## Usage

```bash
# add source PDFs to data/raw/, then start the app
uv run streamlit run src/app.py
```

Open the app in your browser, use the **Ingest** page to load documents into
Qdrant, then use the **Chat** page to ask questions.

## Project structure

```
data/raw/              source PDFs to ingest
src/ingestion.py       business logic: hash check, parse, chunk, embed, upsert in qdrant
src/retrieval.py       business logic: semantic search + generation
src/vector_store.py    wrapper: Qdrant connection and operations
src/app.py             Streamlit entrypoint (multipage nav)
src/pages/chat.py      UI: chat, calls retrieval.py, shows answer + citations
src/pages/ingest.py    UI: ingestion, calls ingestion.py, shows cost + Confirm button
evals/                 evals and golden dataset
doc/                   specs
tests/                 unit tests
```

## Development

```bash
# run tests
uv run pytest
```

## License

TBD
