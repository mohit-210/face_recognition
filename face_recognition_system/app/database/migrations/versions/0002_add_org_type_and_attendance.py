"""add organization type and attendance records

Revision ID: 0002_add_org_type_and_attendance
Revises: 0001_initial
Create Date: 2026-02-20
"""

from alembic import op
import sqlalchemy as sa

revision = "0002_add_org_type_and_attendance"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "companies",
        sa.Column("organization_type", sa.String(length=20), server_default="company", nullable=False),
    )

    op.create_table(
        "attendance_records",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("company_id", sa.Integer(), sa.ForeignKey("companies.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("attendance_date", sa.Date(), nullable=False),
        sa.Column("first_check_in_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_check_out_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("verification_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="present"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("company_id", "user_id", "attendance_date", name="uq_company_user_attendance_date"),
    )


def downgrade() -> None:
    op.drop_table("attendance_records")
    op.drop_column("companies", "organization_type")
