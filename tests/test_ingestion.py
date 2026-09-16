from pathlib import Path
from types import SimpleNamespace

import pytest

from regulations_rag import ingestion
from regulations_rag.config import EmbeddingModelConfig
from regulations_rag.ingestion import Chunk

FAKE_EMBEDDING_CONFIG = EmbeddingModelConfig(
    name="text-embedding-3-small",
    cost_per_1m_tokens=0.02,
    max_tokens=8191,
    tokenizer_name="cl100k_base",
)


def test_compute_file_hash(tmp_path: Path):
    file_path = tmp_path / "doc.pdf"
    file_path.write_bytes(b"hello world")

    import hashlib

    expected = hashlib.sha256(b"hello world").hexdigest()
    assert ingestion.compute_file_hash(file_path) == expected


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("CELEX_02004R1935-20210327_IT_TXT.pdf", "Celex 02004R1935 20210327 It Txt"),
        ("ppwr.pdf", "Ppwr"),
    ],
)
def test_candidate_title_from_filename(filename, expected):
    assert ingestion.candidate_title_from_filename(filename) == expected


def test_estimate_cost():
    chunks = [Chunk(page=1, text="hello world"), Chunk(page=2, text="another chunk of text")]
    total_tokens, total_cost = ingestion.estimate_cost(chunks, FAKE_EMBEDDING_CONFIG)

    assert total_tokens > 0
    assert total_cost == pytest.approx(total_tokens / 1_000_000 * 0.02)


def test_embed_chunks_batches_all_texts():
    chunks = [Chunk(page=1, text="a"), Chunk(page=1, text="b")]
    captured = {}

    def fake_create(model, input):
        captured["model"] = model
        captured["input"] = input
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2]) for _ in input])

    fake_client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))

    vectors = ingestion.embed_chunks(chunks, FAKE_EMBEDDING_CONFIG, fake_client)

    assert vectors == [[0.1, 0.2], [0.1, 0.2]]
    assert captured["model"] == "text-embedding-3-small"
    assert captured["input"] == ["a", "b"]


def test_embed_chunks_empty_list_skips_api_call():
    def fake_create(**kwargs):
        raise AssertionError("should not be called for empty chunks")

    fake_client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))

    assert ingestion.embed_chunks([], FAKE_EMBEDDING_CONFIG, fake_client) == []


def test_ingest_file_upserts_points_with_document_id(mocker):
    fake_vector_store = mocker.Mock()
    mocker.patch(
        "regulations_rag.ingestion.embed_chunks",
        return_value=[[0.1, 0.2], [0.3, 0.4]],
    )

    chunks = [Chunk(page=1, text="one"), Chunk(page=2, text="two")]
    fake_openai_client = object()

    ingestion.ingest_file(
        path=Path("data/raw/ppwr.pdf"),
        title="PPWR",
        chunks=chunks,
        file_hash="deadbeef",
        document_id="doc-123",
        embedding_config=FAKE_EMBEDDING_CONFIG,
        vector_store=fake_vector_store,
        openai_client=fake_openai_client,
    )

    fake_vector_store.upsert_points.assert_called_once()

    points = fake_vector_store.upsert_points.call_args.args[0]
    assert len(points) == 2
    assert points[0].payload == {
        "document_id": "doc-123",
        "title": "PPWR",
        "source_file": "ppwr.pdf",
        "content_hash": "deadbeef",
        "page": 1,
        "text": "one",
        "chunk_number": 0,
    }
    assert points[1].payload["chunk_number"] == 1


def test_ingest_new_document_marks_indexed_on_success(mocker):
    mocker.patch("regulations_rag.ingestion.ingest_file")
    fake_repo = mocker.Mock()

    chunks = [Chunk(page=1, text="one"), Chunk(page=2, text="two")]
    ingestion.ingest_new_document(
        path=Path("data/raw/ppwr.pdf"),
        title="PPWR",
        chunks=chunks,
        file_hash="deadbeef",
        document_id="doc-123",
        embedding_config=FAKE_EMBEDDING_CONFIG,
        vector_store=object(),
        openai_client=object(),
        documents_repo=fake_repo,
    )

    fake_repo.create_pending.assert_called_once_with("doc-123", "ppwr.pdf", "PPWR", "deadbeef")
    fake_repo.mark_indexed.assert_called_once_with("doc-123", 2)
    fake_repo.mark_failed.assert_not_called()


def test_ingest_new_document_marks_failed_on_exception(mocker):
    mocker.patch(
        "regulations_rag.ingestion.ingest_file", side_effect=RuntimeError("embedding blew up")
    )
    fake_repo = mocker.Mock()

    with pytest.raises(RuntimeError):
        ingestion.ingest_new_document(
            path=Path("data/raw/ppwr.pdf"),
            title="PPWR",
            chunks=[Chunk(page=1, text="one")],
            file_hash="deadbeef",
            document_id="doc-123",
            embedding_config=FAKE_EMBEDDING_CONFIG,
            vector_store=object(),
            openai_client=object(),
            documents_repo=fake_repo,
        )

    fake_repo.create_pending.assert_called_once()
    fake_repo.mark_failed.assert_called_once_with("doc-123", "embedding blew up")
    fake_repo.mark_indexed.assert_not_called()


def test_replace_document_writes_chunks_before_updating_postgres_row(mocker):
    call_order = []
    mocker.patch(
        "regulations_rag.ingestion.ingest_file",
        side_effect=lambda **kwargs: call_order.append("ingest_file"),
    )
    fake_repo = mocker.Mock()
    fake_repo.get_by_id.return_value = mocker.Mock()
    fake_repo.update_indexed.side_effect = lambda **kwargs: call_order.append("update_indexed")

    ingestion.replace_document(
        path=Path("data/raw/ppwr.pdf"),
        title="PPWR",
        chunks=[Chunk(page=1, text="one")],
        file_hash="deadbeef",
        document_id="doc-123",
        embedding_config=FAKE_EMBEDDING_CONFIG,
        vector_store=object(),
        openai_client=object(),
        documents_repo=fake_repo,
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
    mocker.patch("regulations_rag.ingestion.ingest_file", side_effect=RuntimeError("boom"))
    fake_repo = mocker.Mock()

    with pytest.raises(RuntimeError):
        ingestion.replace_document(
            path=Path("data/raw/ppwr.pdf"),
            title="PPWR",
            chunks=[Chunk(page=1, text="one")],
            file_hash="deadbeef",
            document_id="doc-123",
            embedding_config=FAKE_EMBEDDING_CONFIG,
            vector_store=object(),
            openai_client=object(),
            documents_repo=fake_repo,
        )

    fake_repo.get_by_id.assert_not_called()
    fake_repo.update_indexed.assert_not_called()
    fake_repo.create_pending.assert_not_called()


def test_replace_document_creates_row_for_legacy_document(mocker):
    mocker.patch("regulations_rag.ingestion.ingest_file")
    fake_repo = mocker.Mock()
    fake_repo.get_by_id.return_value = None

    ingestion.replace_document(
        path=Path("data/raw/ppwr.pdf"),
        title="PPWR",
        chunks=[Chunk(page=1, text="one")],
        file_hash="deadbeef",
        document_id="doc-123",
        embedding_config=FAKE_EMBEDDING_CONFIG,
        vector_store=object(),
        openai_client=object(),
        documents_repo=fake_repo,
    )

    fake_repo.create_pending.assert_called_once_with("doc-123", "ppwr.pdf", "PPWR", "deadbeef")
    fake_repo.mark_indexed.assert_called_once_with("doc-123", 1)
    fake_repo.update_indexed.assert_not_called()
