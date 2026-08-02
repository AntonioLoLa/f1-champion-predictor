"""SQLAlchemy ORM models.

The schema has two halves:

* **Curated F1 data** -- a normalised, provider-agnostic representation of the championship
  (``races``, ``race_results``, ``driver_standings`` ...). Every provider maps onto these
  tables, which is what makes the sources interchangeable.
* **MLOps metadata** -- ingestion runs, dataset versions, training runs, model versions,
  evaluations and predictions, so that any served probability can be traced back to the
  model, the dataset and the commit that produced it.

Engineered feature matrices are deliberately *not* stored here: they are wide, they change
shape whenever a feature block is added, and they are fully reproducible. They live as
Parquet files, with :class:`DatasetVersion` recording the metadata needed to find and
validate them.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = [
    "Base",
    "Circuit",
    "Constructor",
    "ConstructorStanding",
    "DataIngestion",
    "DatasetVersion",
    "Driver",
    "DriverStanding",
    "Evaluation",
    "ModelVersion",
    "PitStop",
    "Prediction",
    "QualifyingResult",
    "Race",
    "RaceResult",
    "SprintResult",
    "TrainingRun",
]

# JSONB on PostgreSQL (indexable, binary) and plain JSON on SQLite, so the same models run
# against the Compose stack and against a dependency-free local database.
JSONType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    """Declarative base for every ORM model."""


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


# ======================================================================================
# Curated Formula 1 data
# ======================================================================================


class Circuit(Base):
    """A racing circuit."""

    __tablename__ = "circuits"

    circuit_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    locality: Mapped[str | None] = mapped_column(String(128))
    country: Mapped[str | None] = mapped_column(String(64))
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)


class Driver(Base):
    """A driver."""

    __tablename__ = "drivers"

    driver_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    code: Mapped[str | None] = mapped_column(String(8))
    permanent_number: Mapped[int | None] = mapped_column(Integer)
    given_name: Mapped[str] = mapped_column(String(64))
    family_name: Mapped[str] = mapped_column(String(64))
    date_of_birth: Mapped[dt.date | None] = mapped_column(Date)
    nationality: Mapped[str | None] = mapped_column(String(64))

    @property
    def full_name(self) -> str:
        """The driver's display name."""
        return f"{self.given_name} {self.family_name}".strip()


class Constructor(Base):
    """A constructor (team)."""

    __tablename__ = "constructors"

    constructor_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    nationality: Mapped[str | None] = mapped_column(String(64))


class Race(Base):
    """A Grand Prix weekend.

    ``has_sprint`` is what allows the remaining-points calculation to be exact: a sprint
    weekend puts more points on the table than a conventional one.
    """

    __tablename__ = "races"
    __table_args__ = (
        UniqueConstraint("season", "round", name="uq_races_season_round"),
        Index("ix_races_season", "season"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    round: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(128))
    date: Mapped[dt.date | None] = mapped_column(Date)
    circuit_id: Mapped[str | None] = mapped_column(ForeignKey("circuits.circuit_id"))
    has_sprint: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    circuit: Mapped[Circuit | None] = relationship(lazy="joined")


class _ResultMixin:
    """Columns shared by Grand Prix and sprint results."""

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    grid: Mapped[int | None] = mapped_column(Integer)
    position: Mapped[int | None] = mapped_column(Integer)
    """Classified finishing position, or NULL if the driver was not classified."""

    position_order: Mapped[int] = mapped_column(Integer, nullable=False)
    """Dense ordering that always has a value, including for retirements."""

    points: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    laps: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str | None] = mapped_column(String(64))
    finished: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    """True when the driver was classified (including lapped finishers)."""


class RaceResult(_ResultMixin, Base):
    """A driver's result in a Grand Prix."""

    __tablename__ = "race_results"
    __table_args__ = (
        UniqueConstraint("race_id", "driver_id", name="uq_race_results_race_driver"),
        Index("ix_race_results_driver", "driver_id"),
    )

    race_id: Mapped[int] = mapped_column(ForeignKey("races.id", ondelete="CASCADE"))
    driver_id: Mapped[str] = mapped_column(ForeignKey("drivers.driver_id"))
    constructor_id: Mapped[str] = mapped_column(ForeignKey("constructors.constructor_id"))
    fastest_lap_rank: Mapped[int | None] = mapped_column(Integer)
    """Rank of this driver's fastest lap; 1 means the bonus point (2019-2024)."""


class SprintResult(_ResultMixin, Base):
    """A driver's result in a sprint race."""

    __tablename__ = "sprint_results"
    __table_args__ = (
        UniqueConstraint("race_id", "driver_id", name="uq_sprint_results_race_driver"),
    )

    race_id: Mapped[int] = mapped_column(ForeignKey("races.id", ondelete="CASCADE"))
    driver_id: Mapped[str] = mapped_column(ForeignKey("drivers.driver_id"))
    constructor_id: Mapped[str] = mapped_column(ForeignKey("constructors.constructor_id"))


class QualifyingResult(Base):
    """A driver's qualifying result.

    Session times are stored in milliseconds because the source formats them as ``m:ss.SSS``
    strings, which are useless for arithmetic.
    """

    __tablename__ = "qualifying_results"
    __table_args__ = (UniqueConstraint("race_id", "driver_id", name="uq_qualifying_race_driver"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.id", ondelete="CASCADE"))
    driver_id: Mapped[str] = mapped_column(ForeignKey("drivers.driver_id"))
    constructor_id: Mapped[str] = mapped_column(ForeignKey("constructors.constructor_id"))
    position: Mapped[int | None] = mapped_column(Integer)
    q1_ms: Mapped[int | None] = mapped_column(Integer)
    q2_ms: Mapped[int | None] = mapped_column(Integer)
    q3_ms: Mapped[int | None] = mapped_column(Integer)


class DriverStanding(Base):
    """Championship standing for a driver after a given round.

    Read from the provider rather than recomputed: until 1990 the championship used
    "best N results" rules, so a driver's total is not the sum of their race points.
    """

    __tablename__ = "driver_standings"
    __table_args__ = (
        UniqueConstraint("race_id", "driver_id", name="uq_driver_standings_race_driver"),
        Index("ix_driver_standings_driver", "driver_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.id", ondelete="CASCADE"))
    driver_id: Mapped[str] = mapped_column(ForeignKey("drivers.driver_id"))
    constructor_id: Mapped[str | None] = mapped_column(ForeignKey("constructors.constructor_id"))
    points: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    position: Mapped[int | None] = mapped_column(Integer)
    wins: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class ConstructorStanding(Base):
    """Championship standing for a constructor after a given round."""

    __tablename__ = "constructor_standings"
    __table_args__ = (
        UniqueConstraint(
            "race_id", "constructor_id", name="uq_constructor_standings_race_constructor"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.id", ondelete="CASCADE"))
    constructor_id: Mapped[str] = mapped_column(ForeignKey("constructors.constructor_id"))
    points: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    position: Mapped[int | None] = mapped_column(Integer)
    wins: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class PitStop(Base):
    """A single pit stop. Available from 2011 onwards only."""

    __tablename__ = "pit_stops"
    __table_args__ = (
        UniqueConstraint("race_id", "driver_id", "stop", name="uq_pit_stops_race_driver_stop"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.id", ondelete="CASCADE"))
    driver_id: Mapped[str] = mapped_column(ForeignKey("drivers.driver_id"))
    stop: Mapped[int] = mapped_column(Integer, nullable=False)
    lap: Mapped[int | None] = mapped_column(Integer)
    duration_ms: Mapped[int | None] = mapped_column(Integer)


# ======================================================================================
# MLOps metadata
# ======================================================================================


class DataIngestion(Base):
    """Audit record for one data acquisition run."""

    __tablename__ = "data_ingestions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    seasons_covered: Mapped[str | None] = mapped_column(String(64))
    rows_written: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    row_counts: Mapped[dict[str, int] | None] = mapped_column(JSONType)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)


class DatasetVersion(Base):
    """Metadata for a built feature matrix stored as Parquet."""

    __tablename__ = "dataset_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    n_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    n_features: Mapped[int] = mapped_column(Integer, nullable=False)
    min_season: Mapped[int] = mapped_column(Integer, nullable=False)
    max_season: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_names: Mapped[list[str]] = mapped_column(JSONType, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    """Hash of the feature configuration, so a stale dataset is detectable."""

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class TrainingRun(Base):
    """One execution of the training pipeline."""

    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_uuid: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64))
    dataset_version_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_versions.id"))
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    git_sha: Mapped[str | None] = mapped_column(String(40))
    config_hash: Mapped[str | None] = mapped_column(String(64))
    params: Mapped[dict[str, object] | None] = mapped_column(JSONType)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)

    dataset_version: Mapped[DatasetVersion | None] = relationship(lazy="joined")


class ModelVersion(Base):
    """A trained, persisted model candidate."""

    __tablename__ = "model_versions"
    __table_args__ = (Index("ix_model_versions_active", "is_active"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    training_run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id"))
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False)
    mlflow_model_uri: Mapped[str | None] = mapped_column(Text)
    mlflow_base_model_uri: Mapped[str | None] = mapped_column(Text)
    """URI of the pre-calibration pipeline, logged separately for explainability."""

    oof_predictions_path: Mapped[str | None] = mapped_column(Text)
    """Parquet of every walk-forward fold's held-out, leakage-free predictions."""

    fit_seasons: Mapped[list[int] | None] = mapped_column(JSONType)
    """Seasons the deployed artifact was actually fitted on.

    Recorded so a served prediction can say whether it is out of sample. The deployed model
    is refit on nearly every completed season, so scoring a *historical* round replays a
    season the model was trained on, and the result flatters it enormously -- on 2007 the
    refit model picks the champion at 94% of rounds where its own walk-forward predictions
    managed 29%. Both numbers are honest about different questions; showing the first
    without saying which one it is, is not.
    """

    calibration_seasons: Mapped[list[int] | None] = mapped_column(JSONType)
    """Seasons held out of the fit to train the probability calibrator."""

    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_best: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    """Whether this candidate won model selection within its training run."""

    metrics: Mapped[dict[str, float] | None] = mapped_column(JSONType)
    params: Mapped[dict[str, object] | None] = mapped_column(JSONType)
    feature_names: Mapped[list[str] | None] = mapped_column(JSONType)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    training_run: Mapped[TrainingRun] = relationship(lazy="joined")


class Evaluation(Base):
    """A single metric value for a model, split and season phase.

    Long format rather than one column per metric: the set of metrics grows, and this shape
    makes "log loss by phase across folds" a trivial query.
    """

    __tablename__ = "evaluations"
    __table_args__ = (Index("ix_evaluations_model_metric", "model_version_id", "metric_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE")
    )
    split: Mapped[str] = mapped_column(String(32), nullable=False)
    """``train``, ``validation`` or ``test``."""

    fold: Mapped[int | None] = mapped_column(Integer)
    season: Mapped[int | None] = mapped_column(Integer)
    phase: Mapped[str | None] = mapped_column(String(16))
    """Season phase (``early``/``mid``/``late``), or NULL for pooled metrics."""

    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_value: Mapped[float | None] = mapped_column(Float)


class Prediction(Base):
    """A stored championship probability for one driver after one round."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint(
            "model_version_id",
            "season",
            "round",
            "driver_id",
            name="uq_predictions_model_season_round_driver",
        ),
        Index("ix_predictions_season_round", "season", "round"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    round: Mapped[int] = mapped_column(Integer, nullable=False)
    driver_id: Mapped[str] = mapped_column(ForeignKey("drivers.driver_id"))
    probability_raw: Mapped[float] = mapped_column(Float, nullable=False)
    probability_calibrated: Mapped[float | None] = mapped_column(Float)
    probability_normalised: Mapped[float | None] = mapped_column(Float)
    """Calibrated probability rescaled so a (season, round) group sums to 1."""

    rank: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
