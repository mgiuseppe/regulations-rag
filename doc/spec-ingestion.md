
## Interface
The business logic lives in `src/ingestion.py` (framework-agnostic, no
Streamlit imports, fully testable on its own). The Streamlit page
(`src/pages/ingest.py`) only collects user input and displays results — it
calls functions from `ingestion.py` and contains no business logic itself.

## Flow
1. Scan `data/raw/` for PDF files (or, if upload is added later, use uploaded files)
2. For each file:
   1. Extract a candidate title from the filename (e.g. "ppwr.pdf" -> "PPWR")
   2. show the title to the user and let them confirm or edit it
   3. Compute SHA-256 hash of file bytes
   4. Query Qdrant: does a chunk with this `source_file` already exist?
      - if yes and hash matches -> skip, show "already up to date"
      - if yes and hash differs -> mark for re-ingestion (delete old chunks first)
      - if no -> mark for ingestion
3. For files marked for ingestion:
   1. Parse with Docling (PdfPipelineOptions, OCR only if needed)
   2. Chunk with HybridChunker + OpenAITokenizer (text-embedding-3-small, max 8191 tokens)
4. BEFORE embedding:
   1. count total tokens across all chunks to embed (tiktoken),
   2. estimate cost using a hardcoded price per token for text-embedding-3-small,
   3. show a summary per file and total in the UI, 
   4. require the user to click a "Confirm" button before calling the OpenAI embeddings API
5. If the user does not confirm: skip embedding entirely for this run
6. If confirmed:
   1. Embed all chunks (batched)
   2. For files being re-ingested, delete their old chunks from Qdrant first
   3. Upsert new chunks with payload: title, source_file, content_hash, page, text

## Payload schema
- title: str          (regulation title, confirmed/edited by the user at ingestion time)
- source_file: str    (filename, used as identity for idempotency)
- content_hash: str   (SHA-256 of file bytes)
- page: int
- text: str           (chunk content)

## Cost estimation UI (example)
ppwr.pdf: 45,000 tokens (~$0.0009)
reach.pdf: 120,000 tokens (~$0.0024)
Total: 165,000 tokens (~$0.0033)
[Confirm] button

## Note
Make the embedding model configurable, with the following attributes of the model config, to support comparing small vs large in evals:
   1. cost-per-token
   2. max-tokens
   3. tokenizer name
