import hashlib
import re
import uuid
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import tiktoken
from openai import OpenAI

from regulations_rag.config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS, EmbeddingModelConfig
from regulations_rag.repositories.documents_repository import (
    STATUS_INDEXED,
    DocumentRecord,
    DocumentsRepository,
)
from regulations_rag.repositories.vector_store import NewChunk, VectorStore
from regulations_rag.services.parse_and_chunk import Chunk

RAW_DATA_DIR = Path("data/raw")

# OpenAI's embeddings endpoint rejects a request above 300_000 tokens total
# (max_tokens_per_request); stay under it with margin for tokenizer differences.
MAX_TOKENS_PER_EMBEDDING_REQUEST = 250_000
# OpenAI also caps the number of inputs per embeddings request at 2048.
MAX_ITEMS_PER_EMBEDDING_REQUEST = 2048


@dataclass
class DocumentMatch:
    document_id: str
    title: str
    source_file: str
    content_hash: str
    matched_on: list[str]


MatchStatus = Literal["exact_match", "ambiguous", "no_match"]


@dataclass
class MatchResult:
    status: MatchStatus
    exact_matches: list[DocumentMatch]
    ambiguous_matches: list[DocumentMatch]


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


@cache
def _get_documents_repo() -> DocumentsRepository:
    return DocumentsRepository()


@cache
def _get_vector_store() -> VectorStore:
    return VectorStore()


@cache
def _get_llm_client() -> OpenAI:
    return OpenAI()


class IngestionService:
    def __init__(
        self,
        documents_repo: DocumentsRepository | None = None,
        vector_store: VectorStore | None = None,
        llm_client: OpenAI | None = None,
        embedding_config: EmbeddingModelConfig | None = None,
    ):
        self._documents_repo = documents_repo or _get_documents_repo()
        self._vector_store = vector_store or _get_vector_store()
        self._llm_client = llm_client or _get_llm_client()
        self._embedding_config = embedding_config or EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]
        self._vector_store.ensure_collection()

    @property
    def embedding_config(self) -> EmbeddingModelConfig:
        return self._embedding_config

    def candidate_title_from_filename(self, filename: str) -> str:
        stem = Path(filename).stem
        stem = re.sub(r"[_\-]+", " ", stem)
        return stem.strip().title()

    def compute_file_hash(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def estimate_cost(self, chunks: list[Chunk]) -> tuple[int, float]:
        encoding = tiktoken.get_encoding(self._embedding_config.tokenizer_name)
        total_tokens = sum(len(encoding.encode(chunk.text)) for chunk in chunks)
        total_cost = total_tokens / 1_000_000 * self._embedding_config.cost_per_1m_tokens
        return total_tokens, total_cost

    def find_matches(self, content_hash: str, source_file: str, title: str) -> MatchResult:
        """Find already-indexed documents that share the hash, filename, or title
        of a candidate upload, and classify the result so the caller can decide
        how to proceed without re-deriving the match logic itself:
        - "exact_match": a document with the same content hash already exists
        - "ambiguous": no hash match, but one or more documents share the
          filename or title — the caller should ask the user how to proceed
        - "no_match": nothing shares any signal — treat the upload as new

        Looks up each signal with its own indexed query rather than scanning
        the whole catalog; a record is only treated as a match if it's
        STATUS_INDEXED (a pending/failed row has no usable chunks in Qdrant
        yet, so it isn't really "in the library").
        """
        candidates: dict[str, DocumentRecord] = {}
        matched_on_by_id: dict[str, list[str]] = {}

        def _register(record: DocumentRecord | None, reason: str) -> None:
            if record is None or record.status != STATUS_INDEXED:
                return
            candidates[record.id] = record
            matched_on_by_id.setdefault(record.id, []).append(reason)

        _register(self._documents_repo.get_by_content_hash(content_hash), "hash")
        _register(self._documents_repo.get_by_filename(source_file), "filename")
        if title.strip():
            _register(self._documents_repo.get_by_title(title), "title")

        matches = [
            DocumentMatch(
                document_id=record.id,
                title=record.title,
                source_file=record.filename,
                content_hash=record.content_hash,
                matched_on=matched_on_by_id[record.id],
            )
            for record in candidates.values()
        ]
        exact_matches = [m for m in matches if "hash" in m.matched_on]
        ambiguous_matches = [m for m in matches if "hash" not in m.matched_on]

        if exact_matches:
            status: MatchStatus = "exact_match"
        elif ambiguous_matches:
            status = "ambiguous"
        else:
            status = "no_match"

        return MatchResult(
            status=status, exact_matches=exact_matches, ambiguous_matches=ambiguous_matches
        )

    def _embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        if not chunks:
            return []

        encoding = tiktoken.get_encoding(self._embedding_config.tokenizer_name)
        vectors: list[list[float]] = []
        for batch in _batch_chunks(chunks, encoding):
            response = self._llm_client.embeddings.create(
                model=self._embedding_config.name,
                input=[chunk.text for chunk in batch],
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def _ingest_file(
        self,
        path: Path,
        title: str,
        chunks: list[Chunk],
        file_hash: str,
        document_id: str,
    ) -> None:
        source_file = path.name
        vectors = self._embed_chunks(chunks)

        new_chunks = [
            NewChunk(
                id=str(uuid.uuid4()),
                vector=vector,
                document_id=document_id,
                title=title,
                source_file=source_file,
                content_hash=file_hash,
                page=chunk.page,
                text=chunk.text,
                chunk_number=chunk_number,
            )
            for chunk_number, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self._vector_store.upsert_points(new_chunks)

    def ingest_new_document(
        self,
        path: Path,
        title: str,
        chunks: list[Chunk],
        file_hash: str,
        document_id: str,
    ) -> DocumentRecord:
        """Ingest a brand-new document: row goes pending -> indexed, or -> failed
        with the error message on any exception. Never left pending.
        """
        self._documents_repo.create_pending(document_id, path.name, title, file_hash)
        try:
            self._ingest_file(
                path=path,
                title=title,
                chunks=chunks,
                file_hash=file_hash,
                document_id=document_id,
            )
        except Exception as exc:
            self._documents_repo.mark_failed(document_id, str(exc))
            raise
        return self._documents_repo.mark_indexed(document_id, len(chunks))

    def replace_document(
        self,
        path: Path,
        title: str,
        chunks: list[Chunk],
        file_hash: str,
        document_id: str,
    ) -> DocumentRecord:
        """Replace an existing document's content, keeping the same document_id.

        Qdrant has no transactions, so the points to remove are snapshotted
        up front, the new chunks are written first, the Postgres row is
        updated only after that succeeds, and the old points are deleted by
        their snapshotted IDs only at the very end — this avoids both a
        window where the document is missing entirely, and deleting the new
        points by mistake (they can share the old points' source_file/title).
        If writing the new chunks fails, the exception propagates and the
        existing Postgres row (and old Qdrant chunks) are left untouched —
        the old document stays valid.
        """
        old_point_ids = self._vector_store.get_document_point_ids(document_id)
        self._ingest_file(
            path=path,
            title=title,
            chunks=chunks,
            file_hash=file_hash,
            document_id=document_id,
        )
        if self._documents_repo.get_by_id(document_id) is None:
            # Document existed only in Qdrant (ingested before Postgres
            # tracking was added) — create its Postgres row now.
            self._documents_repo.create_pending(document_id, path.name, title, file_hash)
            record = self._documents_repo.mark_indexed(document_id, len(chunks))
        else:
            record = self._documents_repo.update_indexed(
                document_id=document_id,
                filename=path.name,
                title=title,
                content_hash=file_hash,
                chunk_count=len(chunks),
            )
        self._vector_store.delete_points(old_point_ids)
        return record

    def delete_document(self, document_id: str) -> None:
        """Delete a document: row goes deleting -> removed, or -> deletion_failed
        with the error message on any exception. Never left deleting without a
        reason. Safe to retry from deletion_failed (or a stuck deleting row)
        since deleting Qdrant chunks by document_id and deleting the Postgres
        row are both no-ops once already done.
        """
        self._documents_repo.mark_deleting(document_id)
        try:
            self._vector_store.delete_document_chunks(document_id)
            self._documents_repo.delete(document_id)
        except Exception as exc:
            self._documents_repo.mark_deletion_failed(document_id, str(exc))
            raise
