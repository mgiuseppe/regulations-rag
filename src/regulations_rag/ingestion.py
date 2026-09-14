import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tiktoken
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from openai import OpenAI
from qdrant_client.models import PointStruct

from regulations_rag import vector_store
from regulations_rag.config import EmbeddingModelConfig

RAW_DATA_DIR = Path("data/raw")

# Below this many characters of extracted text per page, we assume the PDF
# is scanned/image-based and re-parse with OCR enabled.
MIN_CHARS_PER_PAGE_WITHOUT_OCR = 20

# OpenAI's embeddings endpoint rejects a request above 300_000 tokens total
# (max_tokens_per_request); stay under it with margin for tokenizer differences.
MAX_TOKENS_PER_EMBEDDING_REQUEST = 250_000
# OpenAI also caps the number of inputs per embeddings request at 2048.
MAX_ITEMS_PER_EMBEDDING_REQUEST = 2048

IngestStatus = Literal["up_to_date", "stale", "new"]


@dataclass
class Chunk:
    page: int
    text: str


def scan_raw_files(directory: Path = RAW_DATA_DIR) -> list[Path]:
    return sorted(directory.glob("*.pdf"))


def candidate_title_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[_\-]+", " ", stem)
    return stem.strip().title()


def compute_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_status(existing_hash: str | None, new_hash: str) -> IngestStatus:
    if existing_hash is None:
        return "new"
    if existing_hash == new_hash:
        return "up_to_date"
    return "stale"


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
    status: IngestStatus,
    embedding_config: EmbeddingModelConfig,
    qdrant_client,
    openai_client: OpenAI,
) -> None:
    source_file = path.name

    if status == "stale":
        vector_store.delete_by_source_file(qdrant_client, source_file)

    vectors = embed_chunks(chunks, embedding_config, openai_client)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "title": title,
                "source_file": source_file,
                "content_hash": file_hash,
                "page": chunk.page,
                "text": chunk.text,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    vector_store.upsert_points(qdrant_client, points)
