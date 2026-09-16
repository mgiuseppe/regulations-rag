import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from openai import OpenAI
from qdrant_client.models import PointStruct

from regulations_rag.config import EmbeddingModelConfig
from regulations_rag.repositories.documents_repository import DocumentRecord, DocumentsRepository
from regulations_rag.repositories.vector_store import VectorStore

RAW_DATA_DIR = Path("data/raw")

# Below this many characters of extracted text per page, we assume the PDF
# is scanned/image-based and re-parse with OCR enabled.
MIN_CHARS_PER_PAGE_WITHOUT_OCR = 20

# OpenAI's embeddings endpoint rejects a request above 300_000 tokens total
# (max_tokens_per_request); stay under it with margin for tokenizer differences.
MAX_TOKENS_PER_EMBEDDING_REQUEST = 250_000
# OpenAI also caps the number of inputs per embeddings request at 2048.
MAX_ITEMS_PER_EMBEDDING_REQUEST = 2048


@dataclass
class Chunk:
    page: int
    text: str


def candidate_title_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[_\-]+", " ", stem)
    return stem.strip().title()


def compute_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _convert(path: Path, do_ocr: bool):
    pipeline_options = PdfPipelineOptions(do_ocr=do_ocr)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    return converter.convert(str(path)).document


def _needs_ocr(document) -> bool:
    num_pages = document.num_pages()
    if num_pages == 0:
        return False
    total_chars = len(document.export_to_text())
    return (total_chars / num_pages) < MIN_CHARS_PER_PAGE_WITHOUT_OCR


def parse_and_chunk(path: Path, embedding_config: EmbeddingModelConfig) -> list[Chunk]:
    document = _convert(path, do_ocr=False)
    if _needs_ocr(document):
        document = _convert(path, do_ocr=True)

    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.get_encoding(embedding_config.tokenizer_name),
        max_tokens=embedding_config.max_tokens,
    )
    chunker = HybridChunker(tokenizer=tokenizer)

    chunks = []
    for chunk in chunker.chunk(document):
        page = chunk.meta.doc_items[0].prov[0].page_no if chunk.meta.doc_items else 1
        chunks.append(Chunk(page=page, text=chunker.contextualize(chunk)))
    return chunks


def estimate_cost(
    chunks: list[Chunk], embedding_config: EmbeddingModelConfig
) -> tuple[int, float]:
    encoding = tiktoken.get_encoding(embedding_config.tokenizer_name)
    total_tokens = sum(len(encoding.encode(chunk.text)) for chunk in chunks)
    total_cost = total_tokens / 1_000_000 * embedding_config.cost_per_1m_tokens
    return total_tokens, total_cost


def _batch_chunks(
    chunks: list[Chunk], encoding
) -> list[list[Chunk]]:
    batches: list[list[Chunk]] = []
    current_batch: list[Chunk] = []
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = len(encoding.encode(chunk.text))
        exceeds_token_limit = current_batch and (
            current_tokens + chunk_tokens > MAX_TOKENS_PER_EMBEDDING_REQUEST
        )
        exceeds_item_limit = len(current_batch) >= MAX_ITEMS_PER_EMBEDDING_REQUEST
        if exceeds_token_limit or exceeds_item_limit:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(chunk)
        current_tokens += chunk_tokens

    if current_batch:
        batches.append(current_batch)
    return batches


def embed_chunks(
    chunks: list[Chunk], embedding_config: EmbeddingModelConfig, openai_client: OpenAI
) -> list[list[float]]:
    if not chunks:
        return []

    encoding = tiktoken.get_encoding(embedding_config.tokenizer_name)
    vectors: list[list[float]] = []
    for batch in _batch_chunks(chunks, encoding):
        response = openai_client.embeddings.create(
            model=embedding_config.name,
            input=[chunk.text for chunk in batch],
        )
        vectors.extend(item.embedding for item in response.data)
    return vectors


def ingest_file(
    path: Path,
    title: str,
    chunks: list[Chunk],
    file_hash: str,
    document_id: str,
    embedding_config: EmbeddingModelConfig,
    vector_store: VectorStore,
    openai_client: OpenAI,
) -> None:
    source_file = path.name
    vectors = embed_chunks(chunks, embedding_config, openai_client)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "document_id": document_id,
                "title": title,
                "source_file": source_file,
                "content_hash": file_hash,
                "page": chunk.page,
                "text": chunk.text,
                "chunk_number": chunk_number,
            },
        )
        for chunk_number, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    vector_store.upsert_points(points)


def ingest_new_document(
    path: Path,
    title: str,
    chunks: list[Chunk],
    file_hash: str,
    document_id: str,
    embedding_config: EmbeddingModelConfig,
    vector_store: VectorStore,
    openai_client: OpenAI,
    documents_repo: DocumentsRepository,
) -> DocumentRecord:
    """Ingest a brand-new document: row goes pending -> indexed, or -> failed
    with the error message on any exception. Never left pending.
    """
    documents_repo.create_pending(document_id, path.name, title, file_hash)
    try:
        ingest_file(
            path=path,
            title=title,
            chunks=chunks,
            file_hash=file_hash,
            document_id=document_id,
            embedding_config=embedding_config,
            vector_store=vector_store,
            openai_client=openai_client,
        )
    except Exception as exc:
        documents_repo.mark_failed(document_id, str(exc))
        raise
    return documents_repo.mark_indexed(document_id, len(chunks))


def replace_document(
    path: Path,
    title: str,
    chunks: list[Chunk],
    file_hash: str,
    document_id: str,
    embedding_config: EmbeddingModelConfig,
    vector_store: VectorStore,
    openai_client: OpenAI,
    documents_repo: DocumentsRepository,
) -> DocumentRecord:
    """Replace an existing document's content, keeping the same document_id.

    Qdrant has no transactions, so the new chunks are written first; the
    Postgres row is updated only after that succeeds; the caller is then
    responsible for deleting the old Qdrant points, only after this returns
    successfully. If writing the new chunks fails, the exception propagates
    and the existing Postgres row (and old Qdrant chunks) are left untouched
    — the old document stays valid.
    """
    ingest_file(
        path=path,
        title=title,
        chunks=chunks,
        file_hash=file_hash,
        document_id=document_id,
        embedding_config=embedding_config,
        vector_store=vector_store,
        openai_client=openai_client,
    )
    if documents_repo.get_by_id(document_id) is None:
        # Document existed only in Qdrant (ingested before Postgres tracking
        # was added) — create its Postgres row now.
        documents_repo.create_pending(document_id, path.name, title, file_hash)
        return documents_repo.mark_indexed(document_id, len(chunks))
    return documents_repo.update_indexed(
        document_id=document_id,
        filename=path.name,
        title=title,
        content_hash=file_hash,
        chunk_count=len(chunks),
    )
