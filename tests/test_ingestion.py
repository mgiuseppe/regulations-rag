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
    "existing_hash,new_hash,expected",
    [
        (None, "abc", "new"),
        ("abc", "abc", "up_to_date"),
        ("abc", "def", "stale"),
    ],
)
def test_check_status(existing_hash, new_hash, expected):
    assert ingestion.check_status(existing_hash, new_hash) == expected


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


def test_ingest_file_stale_deletes_before_upsert(mocker):
    mock_delete = mocker.patch("regulations_rag.vector_store.delete_by_source_file")
    mock_upsert = mocker.patch("regulations_rag.vector_store.upsert_points")
    mocker.patch(
        "regulations_rag.ingestion.embed_chunks",
        return_value=[[0.1, 0.2], [0.3, 0.4]],
    )

    chunks = [Chunk(page=1, text="one"), Chunk(page=2, text="two")]
    fake_qdrant_client = object()
    fake_openai_client = object()

    ingestion.ingest_file(
        path=Path("data/raw/ppwr.pdf"),
        title="PPWR",
        chunks=chunks,
        file_hash="deadbeef",
        status="stale",
        embedding_config=FAKE_EMBEDDING_CONFIG,
        qdrant_client=fake_qdrant_client,
        openai_client=fake_openai_client,
    )

    mock_delete.assert_called_once_with(fake_qdrant_client, "ppwr.pdf")
    mock_upsert.assert_called_once()

    points = mock_upsert.call_args.args[1]
    assert len(points) == 2
    assert points[0].payload == {
        "title": "PPWR",
        "source_file": "ppwr.pdf",
        "content_hash": "deadbeef",
        "page": 1,
        "text": "one",
    }


def test_ingest_file_new_does_not_delete(mocker):
    mock_delete = mocker.patch("regulations_rag.vector_store.delete_by_source_file")
    mocker.patch("regulations_rag.vector_store.upsert_points")
    mocker.patch("regulations_rag.ingestion.embed_chunks", return_value=[[0.1, 0.2]])

    ingestion.ingest_file(
        path=Path("data/raw/reach.pdf"),
        title="REACH",
        chunks=[Chunk(page=1, text="one")],
        file_hash="cafe",
        status="new",
        embedding_config=FAKE_EMBEDDING_CONFIG,
        qdrant_client=object(),
        openai_client=object(),
    )

    mock_delete.assert_not_called()
