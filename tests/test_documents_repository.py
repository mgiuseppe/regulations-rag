import pytest
from sqlalchemy import create_engine

from regulations_rag.repositories.documents_repository import (
    STATUS_FAILED,
    STATUS_INDEXED,
    STATUS_PENDING,
    Base,
    DocumentMatch,
    DocumentsRepository,
)


@pytest.fixture
def repo():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return DocumentsRepository(engine=engine)


def test_create_pending(repo):
    record = repo.create_pending("id-1", "file.pdf", "Title", "hash-1")

    assert record.id == "id-1"
    assert record.status == STATUS_PENDING
    assert record.chunk_count == 0
    assert record.error_message is None


def test_mark_indexed(repo):
    repo.create_pending("id-1", "file.pdf", "Title", "hash-1")

    record = repo.mark_indexed("id-1", chunk_count=7)

    assert record.status == STATUS_INDEXED
    assert record.chunk_count == 7
    assert record.error_message is None


def test_mark_failed_records_error_message(repo):
    repo.create_pending("id-1", "file.pdf", "Title", "hash-1")

    record = repo.mark_failed("id-1", "boom")

    assert record.status == STATUS_FAILED
    assert record.error_message == "boom"


def test_update_indexed_keeps_id_and_updates_fields(repo):
    repo.create_pending("id-1", "old.pdf", "Old Title", "hash-old")
    repo.mark_indexed("id-1", chunk_count=3)

    record = repo.update_indexed(
        document_id="id-1",
        filename="new.pdf",
        title="New Title",
        content_hash="hash-new",
        chunk_count=9,
    )

    assert record.id == "id-1"
    assert record.filename == "new.pdf"
    assert record.title == "New Title"
    assert record.content_hash == "hash-new"
    assert record.chunk_count == 9
    assert record.status == STATUS_INDEXED


def test_get_by_content_hash(repo):
    repo.create_pending("id-1", "file.pdf", "Title", "hash-1")

    found = repo.get_by_content_hash("hash-1")
    missing = repo.get_by_content_hash("does-not-exist")

    assert found is not None
    assert found.id == "id-1"
    assert missing is None


def test_list_documents_sorts_and_paginates(repo):
    repo.create_pending("id-b", "b.pdf", "Banana", "hash-b")
    repo.create_pending("id-a", "a.pdf", "Apple", "hash-a")
    repo.create_pending("id-c", "c.pdf", "Cherry", "hash-c")

    page1, total = repo.list_documents(page=1, page_size=2, sort_by="title", sort_order="asc")
    page2, _ = repo.list_documents(page=2, page_size=2, sort_by="title", sort_order="asc")

    assert total == 3
    assert [d.title for d in page1] == ["Apple", "Banana"]
    assert [d.title for d in page2] == ["Cherry"]


def test_list_documents_rejects_unknown_sort_field(repo):
    with pytest.raises(ValueError):
        repo.list_documents(sort_by="not_a_field")


def test_find_matches_by_hash_filename_and_title(repo):
    repo.create_pending("id-1", "ppwr.pdf", "PPWR", "abc")
    repo.mark_indexed("id-1", chunk_count=3)

    exact_hash = repo.find_matches("abc", "different.pdf", "Different Title")
    assert exact_hash == [
        DocumentMatch(
            document_id="id-1",
            title="PPWR",
            source_file="ppwr.pdf",
            content_hash="abc",
            matched_on=["hash"],
        )
    ]

    filename_match = repo.find_matches("xyz", "ppwr.pdf", "Different Title")
    assert filename_match[0].matched_on == ["filename"]

    title_match = repo.find_matches("xyz", "different.pdf", "ppwr")
    assert title_match[0].matched_on == ["title"]


def test_find_matches_returns_empty_when_nothing_matches(repo):
    repo.create_pending("id-1", "ppwr.pdf", "PPWR", "abc")
    repo.mark_indexed("id-1", chunk_count=3)

    assert repo.find_matches("xyz", "reach.pdf", "REACH") == []


def test_find_matches_ignores_non_indexed_documents(repo):
    repo.create_pending("id-1", "ppwr.pdf", "PPWR", "abc")

    assert repo.find_matches("abc", "ppwr.pdf", "PPWR") == []
