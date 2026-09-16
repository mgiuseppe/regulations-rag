from dataclasses import dataclass, field
from functools import cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from regulations_rag.config import QDRANT_COLLECTION, QDRANT_HOST, QDRANT_PORT, VECTOR_SIZE

SCROLL_PAGE_SIZE = 256


@dataclass
class DocumentSummary:
    key_field: str
    key: str
    title: str
    source_file: str
    content_hash: str


@dataclass
class DocumentMatch(DocumentSummary):
    matched_on: list[str] = field(default_factory=list)


@dataclass
class ChunkRecord:
    id: str
    document_id: str
    title: str
    source_file: str
    content_hash: str
    page: int
    text: str
    chunk_number: int


@dataclass
class NewChunk:
    id: str
    vector: list[float]
    document_id: str
    title: str
    source_file: str
    content_hash: str
    page: int
    text: str
    chunk_number: int


@dataclass
class SearchResult:
    title: str
    source_file: str
    page: int
    text: str
    score: float


@cache
def _get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


class VectorStore:
    def __init__(self, client: QdrantClient | None = None):
        self._client = client or _get_client()

    def ensure_collection(self) -> None:
        if not self._client.collection_exists(QDRANT_COLLECTION):
            self._client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
        self._client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    def _iter_all_points(self):
        if not self._client.collection_exists(QDRANT_COLLECTION):
            return
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=SCROLL_PAGE_SIZE,
                with_payload=True,
                offset=offset,
            )
            yield from points
            if offset is None:
                break

    def list_documents(self) -> list[DocumentSummary]:
        """Return one entry per distinct ingested document (deduplicated across chunks).

        Documents ingested before `document_id` existed are keyed by `source_file`
        instead, so they remain addressable for matching/replacement.
        """
        documents: dict[str, DocumentSummary] = {}
        for point in self._iter_all_points():
            payload = point.payload or {}
            document_id = payload.get("document_id")
            key_field = "document_id" if document_id else "source_file"
            key = document_id or payload.get("source_file")
            if key is None or key in documents:
                continue
            documents[key] = DocumentSummary(
                key_field=key_field,
                key=key,
                title=payload.get("title"),
                source_file=payload.get("source_file"),
                content_hash=payload.get("content_hash"),
            )
        return list(documents.values())

    def find_matches(self, content_hash: str, source_file: str, title: str) -> list[DocumentMatch]:
        """Find already-ingested documents that share the hash, filename, or title
        of a candidate document, so the caller can ask the user how to proceed.
        """
        matches = []
        normalized_title = title.strip().lower() if title else ""
        for document in self.list_documents():
            matched_on = []
            if document.content_hash == content_hash:
                matched_on.append("hash")
            if document.source_file == source_file:
                matched_on.append("filename")
            doc_title = (document.title or "").strip().lower()
            if normalized_title and doc_title == normalized_title:
                matched_on.append("title")
            if matched_on:
                matches.append(
                    DocumentMatch(
                        key_field=document.key_field,
                        key=document.key,
                        title=document.title,
                        source_file=document.source_file,
                        content_hash=document.content_hash,
                        matched_on=matched_on,
                    )
                )
        return matches

    def get_document_chunks(self, document_id: str) -> list[ChunkRecord]:
        """Return every chunk of a document, ordered by page.

        Qdrant's `order_by` cannot be combined with cursor-based pagination
        (`next_page_offset` is always null once `order_by` is set), so instead
        we page through with the regular ID-based cursor and sort client-side.
        """
        if not self._client.collection_exists(QDRANT_COLLECTION):
            return []
        points = []
        offset = None
        while True:
            page, offset = self._client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                ),
                limit=SCROLL_PAGE_SIZE,
                with_payload=True,
                offset=offset,
            )
            points.extend(page)
            if offset is None:
                break
        points.sort(key=lambda point: (point.payload or {}).get("page", 0))
        return [
            ChunkRecord(
                id=point.id,
                document_id=(point.payload or {}).get("document_id"),
                title=(point.payload or {}).get("title"),
                source_file=(point.payload or {}).get("source_file"),
                content_hash=(point.payload or {}).get("content_hash"),
                page=(point.payload or {}).get("page"),
                text=(point.payload or {}).get("text"),
                chunk_number=(point.payload or {}).get("chunk_number"),
            )
            for point in points
        ]

    def get_document_point_ids(self, key_field: str, key: str) -> list[str]:
        """Point IDs belonging to one document, identified by key_field/key
        (as returned by list_documents/find_matches).

        Callers replacing a document should insert the new points first, then
        delete these specific IDs — not filter-delete by key_field/key, since
        when key_field is "source_file" the new points can share that same
        filename and would otherwise be deleted too.
        """
        if not self._client.collection_exists(QDRANT_COLLECTION):
            return []
        ids = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=Filter(must=[FieldCondition(key=key_field, match=MatchValue(value=key))]),
                limit=SCROLL_PAGE_SIZE,
                with_payload=False,
                offset=offset,
            )
            ids.extend(point.id for point in points)
            if offset is None:
                break
        return ids

    def delete_points(self, point_ids: list[str]) -> None:
        if not point_ids:
            return
        self._client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=PointIdsList(points=point_ids),
        )

    def upsert_points(self, chunks: list[NewChunk]) -> None:
        points = [
            PointStruct(
                id=chunk.id,
                vector=chunk.vector,
                payload={
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "source_file": chunk.source_file,
                    "content_hash": chunk.content_hash,
                    "page": chunk.page,
                    "text": chunk.text,
                    "chunk_number": chunk.chunk_number,
                },
            )
            for chunk in chunks
        ]
        self._client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    def search(self, query_vector: list[float], top_n: int) -> list[SearchResult]:
        if not self._client.collection_exists(QDRANT_COLLECTION):
            return []
        points = self._client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=top_n,
            with_payload=True,
        ).points
        return [
            SearchResult(
                title=point.payload["title"],
                source_file=point.payload["source_file"],
                page=point.payload["page"],
                text=point.payload["text"],
                score=point.score,
            )
            for point in points
        ]
