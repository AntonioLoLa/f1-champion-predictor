"""Record which seasons a model version was fitted and calibrated on.

Needed so a served prediction can report whether it is out of sample: the deployed
artifact is refit on nearly every completed season, so replaying a historical round scores
data the model was trained on.

Revision ID: 8f2c1d40a7b3
Revises: 3a3d1b55bd57
Create Date: 2026-07-30

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "8f2c1d40a7b3"
down_revision = "3a3d1b55bd57"
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    """Add the fit/calibration season columns."""
    op.add_column("model_versions", sa.Column("fit_seasons", _JSON, nullable=True))
    op.add_column("model_versions", sa.Column("calibration_seasons", _JSON, nullable=True))


def downgrade() -> None:
    """Drop the fit/calibration season columns."""
    op.drop_column("model_versions", "calibration_seasons")
    op.drop_column("model_versions", "fit_seasons")
