from types import SimpleNamespace

from regulations_rag.repositories.vector_store import ChunkRecord, DocumentMatch, DocumentSummary, VectorStore


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


def test_list_documents_dedupes_by_document_id():
    points = [
        make_point(
            {
                "document_id": "doc-1",
                "title": "PPWR",
                "source_file": "ppwr.pdf",
                "content_hash": "abc",
            }
        ),
        make_point(
            {
                "document_id": "doc-1",
                "title": "PPWR",
                "source_file": "ppwr.pdf",
                "content_hash": "abc",
            }
        ),
    ]
    store = VectorStore(make_client(points))

    documents = store.list_documents()

    assert documents == [
        DocumentSummary(
            document_id="doc-1",
            title="PPWR",
            source_file="ppwr.pdf",
            content_hash="abc",
        )
    ]


def test_list_documents_skips_points_without_document_id():
    points = [make_point({"title": "REACH", "source_file": "reach.pdf", "content_hash": "cafe"})]
    store = VectorStore(make_client(points))

    assert store.list_documents() == []


def test_find_matches_by_hash_filename_and_title():
    points = [
        make_point(
            {
                "document_id": "doc-1",
                "title": "PPWR",
                "source_file": "ppwr.pdf",
                "content_hash": "abc",
            }
        )
    ]
    store = VectorStore(make_client(points))

    exact_hash = store.find_matches("abc", "different.pdf", "Different Title")
    assert exact_hash == [
        DocumentMatch(
            document_id="doc-1",
            title="PPWR",
            source_file="ppwr.pdf",
            content_hash="abc",
            matched_on=["hash"],
        )
    ]

    filename_match = store.find_matches("xyz", "ppwr.pdf", "Different Title")
    assert filename_match[0].matched_on == ["filename"]

    title_match = store.find_matches("xyz", "different.pdf", "ppwr")
    assert title_match[0].matched_on == ["title"]


def test_find_matches_returns_empty_when_nothing_matches():
    points = [
        make_point(
            {
                "document_id": "doc-1",
                "title": "PPWR",
                "source_file": "ppwr.pdf",
                "content_hash": "abc",
            }
        )
    ]
    store = VectorStore(make_client(points))

    assert store.find_matches("xyz", "reach.pdf", "REACH") == []


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
