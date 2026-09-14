from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from regulations_rag.config import QDRANT_COLLECTION, QDRANT_HOST, QDRANT_PORT, VECTOR_SIZE

SCROLL_PAGE_SIZE = 256


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def _iter_all_points(client: QdrantClient):
    if not client.collection_exists(QDRANT_COLLECTION):
        return
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=SCROLL_PAGE_SIZE,
            with_payload=True,
            offset=offset,
        )
        yield from points
        if offset is None:
            break


def list_documents(client: QdrantClient) -> list[dict]:
    """Return one entry per distinct ingested document (deduplicated across chunks).

    Documents ingested before `document_id` existed are keyed by `source_file`
    instead, so they remain addressable for matching/replacement.
    """
    documents: dict[str, dict] = {}
    for point in _iter_all_points(client):
        payload = point.payload or {}
        document_id = payload.get("document_id")
        key_field = "document_id" if document_id else "source_file"
        key = document_id or payload.get("source_file")
        if key is None or key in documents:
            continue
        documents[key] = {
            "key_field": key_field,
            "key": key,
            "title": payload.get("title"),
            "source_file": payload.get("source_file"),
            "content_hash": payload.get("content_hash"),
        }
    return list(documents.values())


def find_matches(
    client: QdrantClient, content_hash: str, source_file: str, title: str
) -> list[dict]:
    """Find already-ingested documents that share the hash, filename, or title
    of a candidate document, so the caller can ask the user how to proceed.
    """
    matches = []
    normalized_title = title.strip().lower() if title else ""
    for document in list_documents(client):
        matched_on = []
        if document["content_hash"] == content_hash:
            matched_on.append("hash")
        if document["source_file"] == source_file:
            matched_on.append("filename")
        doc_title = (document["title"] or "").strip().lower()
        if normalized_title and doc_title == normalized_title:
            matched_on.append("title")
        if matched_on:
            matches.append({**document, "matched_on": matched_on})
    return matches


def get_document_point_ids(client: QdrantClient, key_field: str, key: str) -> list[str]:
    """Point IDs belonging to one document, identified by key_field/key
    (as returned by list_documents/find_matches).

    Callers replacing a document should insert the new points first, then
    delete these specific IDs — not filter-delete by key_field/key, since
    when key_field is "source_file" the new points can share that same
    filename and would otherwise be deleted too.
    """
    if not client.collection_exists(QDRANT_COLLECTION):
        return []
    ids = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key=key_field, match=MatchValue(value=key))]
            ),
            limit=SCROLL_PAGE_SIZE,
            with_payload=False,
            offset=offset,
        )
        ids.extend(point.id for point in points)
        if offset is None:
            break
    return ids


def delete_points(client: QdrantClient, point_ids: list[str]) -> None:
    if not point_ids:
        return
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=PointIdsList(points=point_ids),
    )


def upsert_points(client: QdrantClient, points: list[PointStruct]) -> None:
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)


def search(client: QdrantClient, query_vector: list[float], top_n: int) -> list[ScoredPoint]:
    if not client.collection_exists(QDRANT_COLLECTION):
        return []
    return client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_n,
        with_payload=True,
    ).points
