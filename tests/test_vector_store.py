from types import SimpleNamespace

from regulations_rag import vector_store


def make_client(points):
    return SimpleNamespace(
        collection_exists=lambda name: True,
        scroll=lambda **kwargs: (points, None),
    )


def make_point(payload):
    return SimpleNamespace(payload=payload)


def test_ensure_collection_creates_document_id_index():
    index_calls = []
    client = SimpleNamespace(
        collection_exists=lambda name: True,
        create_payload_index=lambda **kwargs: index_calls.append(kwargs),
    )

    vector_store.ensure_collection(client)

    assert index_calls == [
        {
            "collection_name": vector_store.QDRANT_COLLECTION,
            "field_name": "document_id",
            "field_schema": vector_store.PayloadSchemaType.KEYWORD,
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
    client = make_client(points)

    documents = vector_store.list_documents(client)

    assert documents == [
        {
            "key_field": "document_id",
            "key": "doc-1",
            "title": "PPWR",
            "source_file": "ppwr.pdf",
            "content_hash": "abc",
        }
    ]


def test_list_documents_falls_back_to_source_file_when_no_document_id():
    points = [make_point({"title": "REACH", "source_file": "reach.pdf", "content_hash": "cafe"})]
    client = make_client(points)

    documents = vector_store.list_documents(client)

    assert documents == [
        {
            "key_field": "source_file",
            "key": "reach.pdf",
            "title": "REACH",
            "source_file": "reach.pdf",
            "content_hash": "cafe",
        }
    ]


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
    client = make_client(points)

    exact_hash = vector_store.find_matches(client, "abc", "different.pdf", "Different Title")
    assert exact_hash == [
        {
            "key_field": "document_id",
            "key": "doc-1",
            "title": "PPWR",
            "source_file": "ppwr.pdf",
            "content_hash": "abc",
            "matched_on": ["hash"],
        }
    ]

    filename_match = vector_store.find_matches(client, "xyz", "ppwr.pdf", "Different Title")
    assert filename_match[0]["matched_on"] == ["filename"]

    title_match = vector_store.find_matches(client, "xyz", "different.pdf", "ppwr")
    assert title_match[0]["matched_on"] == ["title"]


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
    client = make_client(points)

    assert vector_store.find_matches(client, "xyz", "reach.pdf", "REACH") == []


def test_get_document_point_ids_returns_ids_matching_key_field():
    points = [
        SimpleNamespace(id="p1", payload=None),
        SimpleNamespace(id="p2", payload=None),
    ]
    client = SimpleNamespace(
        collection_exists=lambda name: True,
        scroll=lambda **kwargs: (points, None),
    )

    ids = vector_store.get_document_point_ids(client, "document_id", "doc-1")

    assert ids == ["p1", "p2"]


def test_get_document_point_ids_returns_empty_when_collection_missing():
    client = SimpleNamespace(collection_exists=lambda name: False)

    assert vector_store.get_document_point_ids(client, "document_id", "doc-1") == []


def test_delete_points_deletes_by_id(mocker):
    fake_client = SimpleNamespace(delete=mocker.Mock())

    vector_store.delete_points(fake_client, ["p1", "p2"])

    fake_client.delete.assert_called_once()
    kwargs = fake_client.delete.call_args.kwargs
    assert kwargs["collection_name"] == vector_store.QDRANT_COLLECTION


def test_delete_points_skips_call_when_no_ids(mocker):
    fake_client = SimpleNamespace(delete=mocker.Mock())

    vector_store.delete_points(fake_client, [])

    fake_client.delete.assert_not_called()


def test_get_document_chunks_filters_by_document_id_and_sorts_by_page():
    points = [
        make_point({"page": 3, "chunk_number": 4}),
        make_point({"page": 1, "chunk_number": 0}),
        make_point({"page": 1, "chunk_number": 1}),
        make_point({"page": 2, "chunk_number": 2}),
    ]
    scroll_calls = []

    def scroll(**kwargs):
        scroll_calls.append(kwargs)
        return points, None

    client = SimpleNamespace(collection_exists=lambda name: True, scroll=scroll)

    result_points = vector_store.get_document_chunks(client, "doc-1")

    assert [p.payload["page"] for p in result_points] == [1, 1, 2, 3]
    # ties on page keep their original relative order (stable sort)
    assert [p.payload["chunk_number"] for p in result_points] == [0, 1, 2, 4]
    filter_condition = scroll_calls[0]["scroll_filter"].must[0]
    assert filter_condition.key == "document_id"
    assert filter_condition.match.value == "doc-1"
    assert "order_by" not in scroll_calls[0]


def test_get_document_chunks_pages_through_results():
    first_page = [make_point({"page": 1})]
    second_page = [make_point({"page": 2})]
    responses = [(first_page, "cursor-1"), (second_page, None)]

    def scroll(**kwargs):
        return responses.pop(0)

    client = SimpleNamespace(collection_exists=lambda name: True, scroll=scroll)

    result_points = vector_store.get_document_chunks(client, "doc-1")

    assert result_points == first_page + second_page


def test_get_document_chunks_returns_empty_when_collection_missing():
    client = SimpleNamespace(collection_exists=lambda name: False)

    assert vector_store.get_document_chunks(client, "doc-1") == []
