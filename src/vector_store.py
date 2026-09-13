from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.config import QDRANT_COLLECTION, QDRANT_HOST, QDRANT_PORT, VECTOR_SIZE


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def get_existing_document(client: QdrantClient, source_file: str) -> dict | None:
    points, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
        ),
        limit=1,
        with_payload=True,
    )
    if not points:
        return None
    return points[0].payload


def get_existing_hash(client: QdrantClient, source_file: str) -> str | None:
    document = get_existing_document(client, source_file)
    if document is None:
        return None
    return document.get("content_hash")


def delete_by_source_file(client: QdrantClient, source_file: str) -> None:
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
        ),
    )


def upsert_points(client: QdrantClient, points: list[PointStruct]) -> None:
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
