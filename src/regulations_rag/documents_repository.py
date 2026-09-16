"""Postgres-backed catalog of ingested documents.

This module is the only place in the codebase that knows about SQLAlchemy.
Everything else (pages, ingestion logic) exchanges plain `DocumentRecord`
objects with `DocumentsRepository`.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from regulations_rag.config import DATABASE_URL

STATUS_PENDING = "pending"
STATUS_INDEXED = "indexed"
STATUS_FAILED = "failed"

SORTABLE_FIELDS = ("title", "created_at", "file_name")
DEFAULT_SORT_FIELD = "title"


class Base(DeclarativeBase):
    pass


class DocumentModel(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)


@dataclass
class DocumentRecord:
    id: str
    filename: str
    title: str
    content_hash: str
    created_at: datetime
    updated_at: datetime
    chunk_count: int
    status: str
    error_message: str | None


def _to_record(model: DocumentModel) -> DocumentRecord:
    return DocumentRecord(
        id=model.id,
        filename=model.filename,
        title=model.title,
        content_hash=model.content_hash,
        created_at=model.created_at,
        updated_at=model.updated_at,
        chunk_count=model.chunk_count,
        status=model.status,
        error_message=model.error_message,
    )


_engine: Engine | None = None


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine


class DocumentsRepository:
    def __init__(self, engine: Engine | None = None):
        self._engine = engine or _get_engine()

    def _session(self) -> Session:
        return Session(self._engine)

    def create_pending(
        self, document_id: str, filename: str, title: str, content_hash: str
    ) -> DocumentRecord:
        now = datetime.now(timezone.utc)
        model = DocumentModel(
            id=document_id,
            filename=filename,
            title=title,
            content_hash=content_hash,
            created_at=now,
            updated_at=now,
            chunk_count=0,
            status=STATUS_PENDING,
            error_message=None,
        )
        with self._session() as session:
            session.add(model)
            session.commit()
            session.refresh(model)
            return _to_record(model)

    def mark_indexed(self, document_id: str, chunk_count: int) -> DocumentRecord:
        with self._session() as session:
            model = session.get(DocumentModel, document_id)
            if model is None:
                raise ValueError(f"No document with id {document_id!r}")
            model.status = STATUS_INDEXED
            model.chunk_count = chunk_count
            model.error_message = None
            model.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(model)
            return _to_record(model)

    def mark_failed(self, document_id: str, error_message: str) -> DocumentRecord:
        with self._session() as session:
            model = session.get(DocumentModel, document_id)
            if model is None:
                raise ValueError(f"No document with id {document_id!r}")
            model.status = STATUS_FAILED
            model.error_message = error_message
            model.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(model)
            return _to_record(model)

    def update_indexed(
        self,
        document_id: str,
        filename: str,
        title: str,
        content_hash: str,
        chunk_count: int,
    ) -> DocumentRecord:
        """Update an existing document in place, keeping its id.

        Callers replacing a document's Qdrant chunks must call this only
        after the new chunks have been written to Qdrant, and must delete
        the old chunks only after this call succeeds — see ingestion.py.
        """
        with self._session() as session:
            model = session.get(DocumentModel, document_id)
            if model is None:
                raise ValueError(f"No document with id {document_id!r}")
            model.filename = filename
            model.title = title
            model.content_hash = content_hash
            model.chunk_count = chunk_count
            model.status = STATUS_INDEXED
            model.error_message = None
            model.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(model)
            return _to_record(model)

    def get_by_id(self, document_id: str) -> DocumentRecord | None:
        with self._session() as session:
            model = session.get(DocumentModel, document_id)
            return _to_record(model) if model else None

    def get_by_content_hash(self, content_hash: str) -> DocumentRecord | None:
        with self._session() as session:
            model = session.scalar(
                select(DocumentModel).where(DocumentModel.content_hash == content_hash)
            )
            return _to_record(model) if model else None

    def list_documents(
        self,
        page: int = 1,
        page_size: int = 5,
        sort_by: str = DEFAULT_SORT_FIELD,
        sort_order: str = "asc",
    ) -> tuple[list[DocumentRecord], int]:
        if sort_by not in SORTABLE_FIELDS:
            raise ValueError(f"sort_by must be one of {SORTABLE_FIELDS}, got {sort_by!r}")
        if sort_order not in ("asc", "desc"):
            raise ValueError(f"sort_order must be 'asc' or 'desc', got {sort_order!r}")

        column = {
            "title": DocumentModel.title,
            "created_at": DocumentModel.created_at,
            "file_name": DocumentModel.filename,
        }[sort_by]
        order_clause = column.asc() if sort_order == "asc" else column.desc()

        with self._session() as session:
            total = session.scalar(select(func.count()).select_from(DocumentModel)) or 0
            models = session.scalars(
                select(DocumentModel)
                .order_by(order_clause)
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            return [_to_record(m) for m in models], total
