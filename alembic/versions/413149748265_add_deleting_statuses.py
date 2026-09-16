"""add deleting and deletion_failed statuses

Revision ID: 413149748265
Revises: 73ab58f92617
Create Date: 2026-09-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '413149748265'
down_revision: Union[str, Sequence[str], None] = '73ab58f92617'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint('ck_documents_status', 'documents', type_='check')
    op.create_check_constraint(
        'ck_documents_status',
        'documents',
        "status IN ('pending', 'indexed', 'failed', 'deleting', 'deletion_failed')",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('ck_documents_status', 'documents', type_='check')
    op.create_check_constraint(
        'ck_documents_status',
        'documents',
        "status IN ('pending', 'indexed', 'failed')",
    )
