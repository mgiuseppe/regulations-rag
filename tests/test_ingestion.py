from pathlib import Path
from types import SimpleNamespace

import pytest

from regulations_rag.config import EmbeddingModelConfig
from regulations_rag.repositories.documents_repository import STATUS_INDEXED, STATUS_PENDING
from regulations_rag.services import ingestion
from regulations_rag.services.ingestion import DocumentMatch, MatchResult
from regulations_rag.services.parse_and_chunk import Chunk

FAKE_EMBEDDING_CONFIG = EmbeddingModelConfig(
    name="text-embedding-3-small",
    cost_per_1m_tokens=0.02,
    max_tokens=8191,
    tokenizer_name="cl100k_base",
)


def _make_service(
    mocker, documents_repo=None, vector_store=None, llm_client=None, embedding_config=None
):
    documents_repo = documents_repo if documents_repo is not None else mocker.Mock()
    vector_store = vector_store if vector_store is not None else mocker.Mock()
    llm_client = llm_client if llm_client is not None else mocker.Mock()
    embedding_config = embedding_config if embedding_config is not None else FAKE_EMBEDDING_CONFIG
    return ingestion.IngestionService(
        documents_repo=documents_repo,
        vector_store=vector_store,
        llm_client=llm_client,
        embedding_config=embedding_config,
    )


def test_compute_file_hash(tmp_path: Path, mocker):
    file_path = tmp_path / "doc.pdf"
    file_path.write_bytes(b"hello world")

    import hashlib

    expected = hashlib.sha256(b"hello world").hexdigest()
    service = _make_service(mocker)
    assert service.compute_file_hash(file_path) == expected


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("CELEX_02004R1935-20210327_IT_TXT.pdf", "Celex 02004R1935 20210327 It Txt"),
        ("ppwr.pdf", "Ppwr"),
    ],
)
def test_candidate_title_from_filename(filename, expected, mocker):
    service = _make_service(mocker)
    assert service.candidate_title_from_filename(filename) == expected


def test_estimate_cost(mocker):
    chunks = [Chunk(page=1, text="hello world"), Chunk(page=2, text="another chunk of text")]
    service = _make_service(mocker)
    total_tokens, total_cost = service.estimate_cost(chunks)

    assert total_tokens > 0
    assert total_cost == pytest.approx(total_tokens / 1_000_000 * 0.02)


def test_init_ensures_qdrant_collection(mocker):
    fake_vector_store = mocker.Mock()

    _make_service(mocker, vector_store=fake_vector_store)

    fake_vector_store.ensure_collection.assert_called_once()


def test_embed_chunks_batches_all_texts(mocker):
    chunks = [Chunk(page=1, text="a"), Chunk(page=1, text="b")]
    captured = {}

    def fake_create(model, input):
        captured["model"] = model
        captured["input"] = input
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2]) for _ in input])

    fake_client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))
    service = _make_service(mocker, llm_client=fake_client)

    vectors = service._embed_chunks(chunks)

    assert vectors == [[0.1, 0.2], [0.1, 0.2]]
    assert captured["model"] == "text-embedding-3-small"
    assert captured["input"] == ["a", "b"]


def test_embed_chunks_empty_list_skips_api_call(mocker):
    def fake_create(**kwargs):
        raise AssertionError("should not be called for empty chunks")

    fake_client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))
    service = _make_service(mocker, llm_client=fake_client)

    assert service._embed_chunks([]) == []


def test_ingest_file_upserts_points_with_document_id(mocker):
    fake_vector_store = mocker.Mock()
    service = _make_service(mocker, vector_store=fake_vector_store)
    mocker.patch.object(service, "_embed_chunks", return_value=[[0.1, 0.2], [0.3, 0.4]])

    chunks = [Chunk(page=1, text="one"), Chunk(page=2, text="two")]

    service._ingest_file(
        path=Path("data/raw/ppwr.pdf"),
        title="PPWR",
        chunks=chunks,
        file_hash="deadbeef",
        document_id="doc-123",
    )

    fake_vector_store.upsert_points.assert_called_once()

    new_chunks = fake_vector_store.upsert_points.call_args.args[0]
    assert len(new_chunks) == 2
    assert new_chunks[0].document_id == "doc-123"
    assert new_chunks[0].title == "PPWR"
    assert new_chunks[0].source_file == "ppwr.pdf"
    assert new_chunks[0].content_hash == "deadbeef"
    assert new_chunks[0].page == 1
    assert new_chunks[0].text == "one"
    assert new_chunks[0].chunk_number == 0
    assert new_chunks[1].chunk_number == 1


def test_ingest_new_document_marks_indexed_on_success(mocker):
    fake_repo = mocker.Mock()
    service = _make_service(mocker, documents_repo=fake_repo)
    mocker.patch.object(service, "_ingest_file")

    chunks = [Chunk(page=1, text="one"), Chunk(page=2, text="two")]
    service.ingest_new_document(
        path=Path("data/raw/ppwr.pdf"),
        title="PPWR",
        chunks=chunks,
        file_hash="deadbeef",
        document_id="doc-123",
    )

    fake_repo.create_pending.assert_called_once_with("doc-123", "ppwr.pdf", "PPWR", "deadbeef")
    fake_repo.mark_indexed.assert_called_once_with("doc-123", 2)
    fake_repo.mark_failed.assert_not_called()


def test_ingest_new_document_marks_failed_on_exception(mocker):
    fake_repo = mocker.Mock()
    service = _make_service(mocker, documents_repo=fake_repo)
    mocker.patch.object(service, "_ingest_file", side_effect=RuntimeError("embedding blew up"))

    with pytest.raises(RuntimeError):
        service.ingest_new_document(
            path=Path("data/raw/ppwr.pdf"),
            title="PPWR",
            chunks=[Chunk(page=1, text="one")],
            file_hash="deadbeef",
            document_id="doc-123",
        )

    fake_repo.create_pending.assert_called_once()
    fake_repo.mark_failed.assert_called_once_with("doc-123", "embedding blew up")
    fake_repo.mark_indexed.assert_not_called()


def test_replace_document_writes_chunks_before_updating_postgres_row(mocker):
    call_order = []
    fake_repo = mocker.Mock()
    fake_repo.get_by_id.return_value = mocker.Mock()
    fake_repo.update_indexed.side_effect = lambda **kwargs: call_order.append("update_indexed")
    service = _make_service(mocker, documents_repo=fake_repo)
    mocker.patch.object(
        service, "_ingest_file", side_effect=lambda **kwargs: call_order.append("ingest_file")
    )

    service.replace_document(
        path=Path("data/raw/ppwr.pdf"),
        title="PPWR",
        chunks=[Chunk(page=1, text="one")],
        file_hash="deadbeef",
        document_id="doc-123",
    )

    assert call_order == ["ingest_file", "update_indexed"]
    fake_repo.update_indexed.assert_called_once_with(
        document_id="doc-123",
        filename="ppwr.pdf",
        title="PPWR",
        content_hash="deadbeef",
        chunk_count=1,
    )


def test_replace_document_leaves_row_untouched_when_ingest_fails(mocker):
    fake_repo = mocker.Mock()
    service = _make_service(mocker, documents_repo=fake_repo)
    mocker.patch.object(service, "_ingest_file", side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        service.replace_document(
            path=Path("data/raw/ppwr.pdf"),
            title="PPWR",
            chunks=[Chunk(page=1, text="one")],
            file_hash="deadbeef",
            document_id="doc-123",
        )

    fake_repo.get_by_id.assert_not_called()
    fake_repo.update_indexed.assert_not_called()
    fake_repo.create_pending.assert_not_called()


def test_replace_document_creates_row_for_legacy_document(mocker):
    fake_repo = mocker.Mock()
    fake_repo.get_by_id.return_value = None
    service = _make_service(mocker, documents_repo=fake_repo)
    mocker.patch.object(service, "_ingest_file")

    service.replace_document(
        path=Path("data/raw/ppwr.pdf"),
        title="PPWR",
        chunks=[Chunk(page=1, text="one")],
        file_hash="deadbeef",
        document_id="doc-123",
    )

    fake_repo.create_pending.assert_called_once_with("doc-123", "ppwr.pdf", "PPWR", "deadbeef")
    fake_repo.mark_indexed.assert_called_once_with("doc-123", 1)
    fake_repo.update_indexed.assert_not_called()


def test_replace_document_snapshots_and_deletes_old_points_after_insert(mocker):
    call_order = []
    fake_vector_store = mocker.Mock()
    fake_vector_store.get_document_point_ids.side_effect = lambda doc_id: (
        call_order.append("snapshot") or ["old-p1", "old-p2"]
    )
    fake_vector_store.delete_points.side_effect = lambda ids: call_order.append(("delete", ids))
    fake_repo = mocker.Mock()
    fake_repo.get_by_id.return_value = mocker.Mock()
    service = _make_service(mocker, documents_repo=fake_repo, vector_store=fake_vector_store)
    mocker.patch.object(
        service, "_ingest_file", side_effect=lambda **kwargs: call_order.append("ingest_file")
    )

    service.replace_document(
        path=Path("data/raw/ppwr.pdf"),
        title="PPWR",
        chunks=[Chunk(page=1, text="one")],
        file_hash="deadbeef",
        document_id="doc-123",
    )

    assert call_order == ["snapshot", "ingest_file", ("delete", ["old-p1", "old-p2"])]
    fake_vector_store.get_document_point_ids.assert_called_once_with("doc-123")


def test_delete_document_marks_deleting_then_removes(mocker):
    fake_repo = mocker.Mock()
    fake_vector_store = mocker.Mock()
    service = _make_service(mocker, documents_repo=fake_repo, vector_store=fake_vector_store)

    service.delete_document("doc-123")

    fake_repo.mark_deleting.assert_called_once_with("doc-123")
    fake_vector_store.delete_document_chunks.assert_called_once_with("doc-123")
    fake_repo.delete.assert_called_once_with("doc-123")
    fake_repo.mark_deletion_failed.assert_not_called()


def test_delete_document_marks_deletion_failed_on_exception(mocker):
    fake_repo = mocker.Mock()
    fake_vector_store = mocker.Mock()
    fake_vector_store.delete_document_chunks.side_effect = RuntimeError("qdrant down")
    service = _make_service(mocker, documents_repo=fake_repo, vector_store=fake_vector_store)

    with pytest.raises(RuntimeError):
        service.delete_document("doc-123")

    fake_repo.mark_deleting.assert_called_once_with("doc-123")
    fake_repo.mark_deletion_failed.assert_called_once_with("doc-123", "qdrant down")
    fake_repo.delete.assert_not_called()


def _record(id, title, filename, content_hash, status=STATUS_INDEXED):
    return SimpleNamespace(id=id, title=title, filename=filename, content_hash=content_hash, status=status)


def test_find_matches_hash_match_is_exact(mocker):
    fake_repo = mocker.Mock()
    fake_repo.get_by_content_hash.return_value = _record("id-1", "PPWR", "ppwr.pdf", "abc")
    fake_repo.get_by_filename.return_value = None
    fake_repo.get_by_title.return_value = None
    service = _make_service(mocker, documents_repo=fake_repo)

    result = service.find_matches("abc", "different.pdf", "Different Title")

    assert result == MatchResult(
        status="exact_match",
        exact_matches=[
            DocumentMatch(
                document_id="id-1",
                title="PPWR",
                source_file="ppwr.pdf",
                content_hash="abc",
                matched_on=["hash"],
            )
        ],
        ambiguous_matches=[],
    )
    fake_repo.get_by_content_hash.assert_called_once_with("abc")
    fake_repo.get_by_filename.assert_called_once_with("different.pdf")
    fake_repo.get_by_title.assert_called_once_with("Different Title")


def test_find_matches_filename_or_title_match_is_ambiguous(mocker):
    fake_repo = mocker.Mock()
    fake_repo.get_by_content_hash.return_value = None
    fake_repo.get_by_filename.return_value = _record("id-1", "PPWR", "ppwr.pdf", "abc")
    fake_repo.get_by_title.return_value = None
    service = _make_service(mocker, documents_repo=fake_repo)

    result = service.find_matches("xyz", "ppwr.pdf", "Different Title")

    assert result.status == "ambiguous"
    assert result.exact_matches == []
    assert [m.matched_on for m in result.ambiguous_matches] == [["filename"]]


def test_find_matches_merges_matched_on_for_the_same_document(mocker):
    same_doc = _record("id-1", "PPWR", "ppwr.pdf", "abc")
    fake_repo = mocker.Mock()
    fake_repo.get_by_content_hash.return_value = same_doc
    fake_repo.get_by_filename.return_value = same_doc
    fake_repo.get_by_title.return_value = None
    service = _make_service(mocker, documents_repo=fake_repo)

    result = service.find_matches("abc", "ppwr.pdf", "Different Title")

    assert result.status == "exact_match"
    assert len(result.exact_matches) == 1
    assert result.exact_matches[0].matched_on == ["hash", "filename"]


def test_find_matches_returns_no_match_when_nothing_matches(mocker):
    fake_repo = mocker.Mock()
    fake_repo.get_by_content_hash.return_value = None
    fake_repo.get_by_filename.return_value = None
    fake_repo.get_by_title.return_value = None
    service = _make_service(mocker, documents_repo=fake_repo)

    result = service.find_matches("xyz", "reach.pdf", "REACH")

    assert result == MatchResult(status="no_match", exact_matches=[], ambiguous_matches=[])


def test_find_matches_ignores_non_indexed_documents(mocker):
    fake_repo = mocker.Mock()
    fake_repo.get_by_content_hash.return_value = _record(
        "id-1", "PPWR", "ppwr.pdf", "abc", status=STATUS_PENDING
    )
    fake_repo.get_by_filename.return_value = None
    fake_repo.get_by_title.return_value = None
    service = _make_service(mocker, documents_repo=fake_repo)

    result = service.find_matches("abc", "ppwr.pdf", "PPWR")

    assert result.status == "no_match"


def test_find_matches_skips_title_lookup_when_title_is_blank(mocker):
    fake_repo = mocker.Mock()
    fake_repo.get_by_content_hash.return_value = None
    fake_repo.get_by_filename.return_value = None
    service = _make_service(mocker, documents_repo=fake_repo)

    service.find_matches("xyz", "reach.pdf", "  ")

    fake_repo.get_by_title.assert_not_called()
