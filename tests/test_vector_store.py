from types import SimpleNamespace

from regulations_rag.repositories.vector_store import ChunkRecord, VectorStore


def make_client(points):
    return SimpleNamespace(
        collection_exists=lambda name: True,
        scroll=lambda **kwargs: (points, None),
    )


def make_point(payload):
    return SimpleNamespace(payload=payload)


def test_ensure_collection_creates_document_id_index():
    from regulations_rag.repositories import vector_store as vector_store_module

    index_calls = []
    client = SimpleNamespace(
        collection_exists=lambda name: True,
        create_payload_index=lambda **kwargs: index_calls.append(kwargs),
    )

    VectorStore(client).ensure_collection()

    assert index_calls == [
        {
            "collection_name": vector_store_module.QDRANT_COLLECTION,
            "field_name": "document_id",
            "field_schema": vector_store_module.PayloadSchemaType.KEYWORD,
        }
    ]


def test_get_document_point_ids_returns_matching_ids():
    points = [
        SimpleNamespace(id="p1", payload=None),
        SimpleNamespace(id="p2", payload=None),
    ]
    client = SimpleNamespace(
        collection_exists=lambda name: True,
        scroll=lambda **kwargs: (points, None),
    )
    store = VectorStore(client)

    ids = store.get_document_point_ids("doc-1")

    assert ids == ["p1", "p2"]


def test_get_document_point_ids_returns_empty_when_collection_missing():
    client = SimpleNamespace(collection_exists=lambda name: False)
    store = VectorStore(client)

    assert store.get_document_point_ids("doc-1") == []


def test_delete_points_deletes_by_id(mocker):
    from regulations_rag.repositories import vector_store as vector_store_module

    fake_client = SimpleNamespace(delete=mocker.Mock())
    store = VectorStore(fake_client)

    store.delete_points(["p1", "p2"])

    fake_client.delete.assert_called_once()
    kwargs = fake_client.delete.call_args.kwargs
    assert kwargs["collection_name"] == vector_store_module.QDRANT_COLLECTION


def test_delete_points_skips_call_when_no_ids(mocker):
    fake_client = SimpleNamespace(delete=mocker.Mock())
    store = VectorStore(fake_client)

    store.delete_points([])

    fake_client.delete.assert_not_called()


def test_get_document_chunks_filters_by_document_id_and_sorts_by_page():
    points = [
        make_point({"page": 3, "chunk_number": 4}),
        make_point({"page": 1, "chunk_number": 0}),
        make_point({"page": 1, "chunk_number": 1}),
        make_point({"page": 2, "chunk_number": 2}),
    ]
    for i, point in enumerate(points):
        point.id = f"p{i}"
    scroll_calls = []

    def scroll(**kwargs):
        scroll_calls.append(kwargs)
        return points, None

    client = SimpleNamespace(collection_exists=lambda name: True, scroll=scroll)
    store = VectorStore(client)

    result_chunks = store.get_document_chunks("doc-1")

    assert [c.page for c in result_chunks] == [1, 1, 2, 3]
    # ties on page keep their original relative order (stable sort)
    assert [c.chunk_number for c in result_chunks] == [0, 1, 2, 4]
    filter_condition = scroll_calls[0]["scroll_filter"].must[0]
    assert filter_condition.key == "document_id"
    assert filter_condition.match.value == "doc-1"
    assert "order_by" not in scroll_calls[0]


def test_get_document_chunks_pages_through_results():
    first_page = [make_point({"page": 1})]
    second_page = [make_point({"page": 2})]
    for point in first_page + second_page:
        point.id = "p1"
    responses = [(first_page, "cursor-1"), (second_page, None)]

    def scroll(**kwargs):
        return responses.pop(0)

    client = SimpleNamespace(collection_exists=lambda name: True, scroll=scroll)
    store = VectorStore(client)

    result_chunks = store.get_document_chunks("doc-1")

    assert [c.page for c in result_chunks] == [1, 2]


def test_get_document_chunks_returns_empty_when_collection_missing():
    client = SimpleNamespace(collection_exists=lambda name: False)
    store = VectorStore(client)

    assert store.get_document_chunks("doc-1") == []
