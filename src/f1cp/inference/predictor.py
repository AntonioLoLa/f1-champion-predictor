"""Live championship-probability inference.

Runs the exact feature pipeline that built the training dataset against the database's
*current* state, scores the result with a served model loaded from MLflow, and rescales
probabilities within each ``(season, round)`` group -- the same
:class:`~f1cp.models.normalizer.GroupProbabilityNormalizer` training uses -- so the answer
is "who wins the title", not twenty independent "is this driver *a* champion" guesses.

The engineered frame is deliberately rebuilt from the database on every call rather than
read back from a :class:`~f1cp.db.models.DatasetVersion` parquet: that parquet is a
snapshot from the last ``dataset build``, which may be several rounds stale, while
inference must always reflect whatever has actually been ingested.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlflow
import mlflow.sklearn
import polars as pl

from f1cp.dataset.loader import load_source_frames, source_data_revision
from f1cp.domain.elimination import TitleContention, title_contention
from f1cp.explain.shap_explainer import explain as explain_shap
from f1cp.features.pipeline import run_pipeline
from f1cp.logging_config import get_logger
from f1cp.models.normalizer import GroupProbabilityNormalizer

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

    from f1cp.config.settings import Settings
    from f1cp.db.models import ModelVersion
    from f1cp.db.uow import UnitOfWork

__all__ = [
    "DriverPrediction",
    "ExplainResult",
    "PredictionService",
    "RoundPredictions",
    "TitleContentionResult",
]

logger = get_logger(__name__)

# Two is enough to cover the dashboard's real access pattern -- the latest season, plus
# whichever historical season the user is browsing -- without holding more panels in
# memory than a served process should.
_PANEL_CACHE_SIZE = 2

# Explanations are a few hundred floats each, so this can be generous where panels cannot:
# enough to hold every model version of one training run, explained at one round.
_EXPLAIN_CACHE_SIZE = 16

# (source data revision, model version, season, round)
type _ExplainKey = tuple[tuple[int, ...], int, int, int]

_PROBABILITY_COLUMN = "probability"
_RAW_COLUMN = "probability_raw"
_NORMALIZED_COLUMN = f"{_PROBABILITY_COLUMN}_normalized"
_RANK_COLUMN = "rank"


@dataclass(frozen=True, slots=True)
class DriverPrediction:
    """One driver's championship probability after a round.

    Attributes:
        driver_id: The driver's identifier.
        driver_name: Display name, resolved from the drivers table.
        constructor_id: The team the driver was racing for at this round.
        probability: Calibrated probability, rescaled to sum to 1 within the round.
        probability_calibrated: Calibrated probability before rescaling.
        probability_raw: Pre-calibration model output.
        rank: 1-based rank within the round, best-placed first.
    """

    driver_id: str
    driver_name: str
    constructor_id: str | None
    probability: float
    probability_calibrated: float
    probability_raw: float
    rank: int


@dataclass(frozen=True, slots=True)
class RoundPredictions:
    """Every driver's ranked probability for one round.

    Attributes:
        season: Championship year.
        round: Round within the season.
        model_version_id: The model version that produced these probabilities.
        model_name: Name of that model version.
        drivers: Ranked predictions, best-placed first.
    """

    season: int
    round: int
    model_version_id: int
    model_name: str
    drivers: list[DriverPrediction]


@dataclass(frozen=True, slots=True)
class ExplainResult:
    """Feature attribution for a model's predictions at one round.

    Attributes:
        season: Championship year.
        round: Round within the season.
        model_version_id: The model version explained.
        feature_names: Feature names, most important first.
        importances: Mean absolute SHAP value per feature, same order.
    """

    season: int
    round: int
    model_version_id: int
    feature_names: list[str]
    importances: list[float]


@dataclass(frozen=True, slots=True)
class TitleContentionResult:
    """Which drivers can still mathematically win the title, as of one round.

    Attributes:
        season: Championship year.
        round: Round within the season this was computed after.
        contenders: One entry per driver with a standing, ranked by current points.
    """

    season: int
    round: int
    contenders: list[TitleContention]


class PredictionService:
    """Builds live feature rows and scores them against a served model.

    Holds three small in-process caches -- MLflow models keyed by their URI, engineered
    panels and SHAP explanations keyed by the source data they were derived from -- so
    callers should build a single instance and reuse it for the lifetime of the
    application (see :meth:`f1cp.container.AppContext.predictor`). Everything keyed on
    :func:`~f1cp.dataset.loader.source_data_revision` is re-checked against the database
    on every call, so newly ingested data is never served from a cache.
    """

    def __init__(self, settings: Settings, *, panel_cache_size: int = _PANEL_CACHE_SIZE) -> None:
        """Configure the service.

        Args:
            settings: Application settings.
            panel_cache_size: How many engineered panels to keep in memory at once. Each
                is one ``max_season`` bound and costs tens of megabytes against the full
                database, so the default is deliberately small; pass ``0`` to disable
                panel caching entirely.
        """
        self._settings = settings
        self._pipeline_cache: dict[str, Pipeline] = {}
        self._panel_cache: OrderedDict[tuple[tuple[int, ...], int], pl.DataFrame] = OrderedDict()
        self._panel_cache_size = panel_cache_size
        self._explain_cache: OrderedDict[_ExplainKey, ExplainResult] = OrderedDict()
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

    def predict_round(
        self,
        uow: UnitOfWork,
        *,
        season: int | None = None,
        round_number: int | None = None,
        model_version_id: int | None = None,
        persist: bool = True,
    ) -> RoundPredictions:
        """Score every driver contesting one round.

        Args:
            uow: Open unit of work.
            season: Championship year. Defaults to the season of the latest race with
                stored results.
            round_number: Round within the season. Defaults to that season's latest round
                with stored results.
            model_version_id: Model version to score with. Defaults to the active one.
            persist: Whether to upsert the scored rows into the ``predictions`` table.

        Returns:
            Ranked probabilities for that round.

        Raises:
            ValueError: If no model version is available, no season/round has results, or
                the engineered frame has no rows for the resolved ``(season, round)``.
        """
        model_version = _resolve_model_version(uow, model_version_id)
        resolved_season, resolved_round = _resolve_season_round(uow, season, round_number)

        frame = self._engineered_frame(uow, max_season=resolved_season)
        round_frame = frame.filter(
            (pl.col("season") == resolved_season) & (pl.col("round") == resolved_round)
        )
        if not round_frame.height:
            msg = f"No engineered rows for season {resolved_season} round {resolved_round}"
            raise ValueError(msg)

        scored = self._score(round_frame, model_version)
        if persist:
            self._persist(uow, scored, model_version.id)

        return _to_round_predictions(scored, resolved_season, resolved_round, model_version)

    def predict_season(
        self,
        uow: UnitOfWork,
        season: int,
        *,
        model_version_id: int | None = None,
        persist: bool = True,
    ) -> list[RoundPredictions]:
        """Score every round of a season that already has stored results.

        Args:
            uow: Open unit of work.
            season: Championship year.
            model_version_id: Model version to score with. Defaults to the active one.
            persist: Whether to upsert the scored rows into the ``predictions`` table.

        Returns:
            One :class:`RoundPredictions` per round with stored results, in round order.

        Raises:
            ValueError: If no model version is available or the season has no results.
        """
        model_version = _resolve_model_version(uow, model_version_id)
        rounds = list(uow.races.rounds_with_results(season))
        if not rounds:
            msg = f"No results stored for season {season}"
            raise ValueError(msg)

        frame = self._engineered_frame(uow, max_season=season)
        season_frame = frame.filter(pl.col("season") == season)
        scored = self._score(season_frame, model_version)
        if persist:
            self._persist(uow, scored, model_version.id)

        return [
            _to_round_predictions(
                scored.filter(pl.col("round") == round_number),
                season,
                round_number,
                model_version,
            )
            for round_number in rounds
        ]

    def explain_round(
        self,
        uow: UnitOfWork,
        *,
        season: int | None = None,
        round_number: int | None = None,
        model_version_id: int | None = None,
    ) -> ExplainResult:
        """Explain a model's inputs for one round via SHAP.

        Args:
            uow: Open unit of work.
            season: Championship year. Defaults to the season of the latest race with
                stored results.
            round_number: Round within the season. Defaults to that season's latest round
                with stored results.
            model_version_id: Model version to explain. Defaults to the active one.

        Memoised per ``(source data, model version, season, round)``. The computation is
        deterministic given the configured random seed, and for algorithms with no fast
        native explainer it costs tens of seconds to reproduce.

        Returns:
            Feature importances for the drivers contesting that round.

        Raises:
            ValueError: If no model version is available, it has no logged base pipeline,
                or the engineered frame has no rows for the resolved ``(season, round)``.
        """
        model_version = _resolve_model_version(uow, model_version_id)
        resolved_season, resolved_round = _resolve_season_round(uow, season, round_number)

        if not model_version.mlflow_base_model_uri:
            msg = f"Model version {model_version.id} has no logged base pipeline to explain"
            raise ValueError(msg)

        cache_key = (
            source_data_revision(uow),
            model_version.id,
            resolved_season,
            resolved_round,
        )
        if (memoised := self._explain_cache.get(cache_key)) is not None:
            self._explain_cache.move_to_end(cache_key)
            return memoised

        frame = self._engineered_frame(uow, max_season=resolved_season)
        round_frame = frame.filter(
            (pl.col("season") == resolved_season) & (pl.col("round") == resolved_round)
        )
        if not round_frame.height:
            msg = f"No engineered rows for season {resolved_season} round {resolved_round}"
            raise ValueError(msg)

        feature_names = _feature_names(model_version, round_frame)
        x = round_frame.select(feature_names).cast(pl.Float64).to_numpy()
        base_pipeline = self._load_pipeline(model_version.mlflow_base_model_uri)

        shap_config = self._settings.explainability.shap
        result = explain_shap(
            base_pipeline,
            model_version.algorithm,
            x,
            feature_names,
            shap_config,
            random_seed=self._settings.app.random_seed,
        )
        explained = ExplainResult(
            season=resolved_season,
            round=resolved_round,
            model_version_id=model_version.id,
            feature_names=result.feature_names,
            importances=result.importances,
        )

        for stale in [key for key in self._explain_cache if key[0] != cache_key[0]]:
            del self._explain_cache[stale]
        self._explain_cache[cache_key] = explained
        while len(self._explain_cache) > _EXPLAIN_CACHE_SIZE:
            self._explain_cache.popitem(last=False)
        return explained

    def title_contention(
        self,
        uow: UnitOfWork,
        *,
        season: int | None = None,
        round_number: int | None = None,
    ) -> TitleContentionResult:
        """Determine which drivers can still mathematically win the title.

        Pure combinatorial reasoning over the points table and the remaining calendar --
        no model, no features, no MLflow -- see :mod:`f1cp.domain.elimination` for why this
        is a closed-form adversarial bound rather than a search problem.

        Args:
            uow: Open unit of work.
            season: Championship year. Defaults to the season of the latest race with
                stored results.
            round_number: Round within the season. Defaults to that season's latest round
                with stored results.

        Returns:
            Every driver with a standing at that round, ranked by current points.

        Raises:
            ValueError: If no season/round with stored standings can be resolved.
        """
        resolved_season, resolved_round = _resolve_season_round(uow, season, round_number)

        standings = uow.standings.for_round(resolved_season, resolved_round)
        remaining = uow.races.remaining(resolved_season, after_round=resolved_round)

        contenders = title_contention(
            {row.driver_id: row.points for row in standings},
            season=resolved_season,
            remaining_rounds=[race.has_sprint for race in remaining],
        )
        return TitleContentionResult(
            season=resolved_season, round=resolved_round, contenders=contenders
        )

    def _engineered_frame(self, uow: UnitOfWork, *, max_season: int) -> pl.DataFrame:
        """The full engineered feature panel up to ``max_season``, with driver names.

        Rebuilding the panel is by far the most expensive thing a prediction does, and it
        depends only on the source data -- not on which model is being served. Caching it
        against :func:`~f1cp.dataset.loader.source_data_revision` is therefore what makes
        switching model versions, or moving between dashboard pages, cost a rescore rather
        than a full rebuild, *without* weakening the freshness guarantee this module's
        docstring makes: the revision is re-read on every call, and any change to the
        source data misses the cache and rebuilds from scratch.
        """
        revision = source_data_revision(uow)
        key = (revision, max_season)
        if (cached := self._panel_cache.get(key)) is not None:
            self._panel_cache.move_to_end(key)
            return cached

        sources = load_source_frames(uow, max_season=max_season)
        frame = run_pipeline(sources, self._settings.features, self._settings.paths)
        names = sources.drivers.select(
            "driver_id",
            (pl.col("given_name") + pl.lit(" ") + pl.col("family_name")).alias("driver_name"),
        )
        panel = frame.join(names, on="driver_id", how="left")

        if self._panel_cache_size > 0:
            # Panels built from superseded data can never be served again, so drop them
            # rather than let them occupy a slot until they age out of the LRU.
            for stale in [k for k in self._panel_cache if k[0] != revision]:
                del self._panel_cache[stale]
            self._panel_cache[key] = panel
            while len(self._panel_cache) > self._panel_cache_size:
                self._panel_cache.popitem(last=False)
        return panel

    def _score(self, frame: pl.DataFrame, model_version: ModelVersion) -> pl.DataFrame:
        """Score a frame with the model's calibrated (and, if logged, raw) pipeline."""
        if not model_version.mlflow_model_uri:
            msg = f"Model version {model_version.id} has no logged model to score"
            raise ValueError(msg)

        feature_names = _feature_names(model_version, frame)
        x = frame.select(feature_names).cast(pl.Float64).to_numpy()

        calibrated_pipeline = self._load_pipeline(model_version.mlflow_model_uri)
        calibrated = calibrated_pipeline.predict_proba(x)[:, 1]

        if model_version.mlflow_base_model_uri and (
            model_version.mlflow_base_model_uri != model_version.mlflow_model_uri
        ):
            base_pipeline = self._load_pipeline(model_version.mlflow_base_model_uri)
            raw = base_pipeline.predict_proba(x)[:, 1]
        else:
            raw = calibrated

        scored = frame.with_columns(
            pl.Series(_PROBABILITY_COLUMN, calibrated),
            pl.Series(_RAW_COLUMN, raw),
        )
        group_columns = self._settings.training.normalisation.group_columns
        normalized = GroupProbabilityNormalizer(group_columns).normalize(
            scored, _PROBABILITY_COLUMN
        )
        return normalized.with_columns(
            pl.col(_NORMALIZED_COLUMN)
            .rank(method="ordinal", descending=True)
            .over(group_columns)
            .cast(pl.Int64)
            .alias(_RANK_COLUMN)
        )

    def _load_pipeline(self, uri: str) -> Pipeline:
        """Load (and cache) a pipeline logged to MLflow."""
        if uri not in self._pipeline_cache:
            logger.info("Loading model from MLflow", extra={"uri": uri})
            self._pipeline_cache[uri] = mlflow.sklearn.load_model(uri)
        return self._pipeline_cache[uri]

    def _persist(self, uow: UnitOfWork, scored: pl.DataFrame, model_version_id: int) -> None:
        """Upsert scored rows into the ``predictions`` table for traceability."""
        rows = [
            {
                "model_version_id": model_version_id,
                "season": row["season"],
                "round": row["round"],
                "driver_id": row["driver_id"],
                "probability_raw": row[_RAW_COLUMN],
                "probability_calibrated": row[_PROBABILITY_COLUMN],
                "probability_normalised": row[_NORMALIZED_COLUMN],
                "rank": row[_RANK_COLUMN],
            }
            for row in scored.select(
                "season",
                "round",
                "driver_id",
                _RAW_COLUMN,
                _PROBABILITY_COLUMN,
                _NORMALIZED_COLUMN,
                _RANK_COLUMN,
            ).to_dicts()
        ]
        uow.predictions.upsert_many(rows)


def _feature_names(model_version: ModelVersion, frame: pl.DataFrame) -> list[str]:
    """The model's training-time feature columns, validated against the live frame.

    Args:
        model_version: The model version being scored or explained.
        frame: The freshly engineered frame the model is about to see.

    Returns:
        The feature column names, in training order.

    Raises:
        ValueError: If the current feature configuration no longer produces one or more
            columns the model was trained on -- a feature block was toggled or renamed
            since training, and the model needs to be retrained before it can serve again.
    """
    feature_names = list(model_version.feature_names or [])
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        msg = (
            f"Model version {model_version.id} was trained on features no longer produced "
            f"by the current feature configuration: {missing}. Retrain with `f1cp train run`."
        )
        raise ValueError(msg)
    return feature_names


def _resolve_model_version(uow: UnitOfWork, model_version_id: int | None) -> ModelVersion:
    """Look up the model version to serve, defaulting to the active one."""
    version = (
        uow.models.get(model_version_id)
        if model_version_id is not None
        else uow.models.get_active()
    )
    if version is None:
        target = model_version_id if model_version_id is not None else "active"
        msg = f"No model version found ({target}); run `f1cp train run` first"
        raise ValueError(msg)
    return version


def _resolve_season_round(
    uow: UnitOfWork, season: int | None, round_number: int | None
) -> tuple[int, int]:
    """Fill in an unspecified season/round with the latest one that has results."""
    if season is not None and round_number is not None:
        return season, round_number

    if season is not None:
        rounds = uow.races.rounds_with_results(season)
        if not rounds:
            msg = f"No results stored for season {season}"
            raise ValueError(msg)
        return season, max(rounds)

    latest = uow.races.latest_completed()
    if latest is None:
        msg = "No race results stored yet; run `f1cp data update --bootstrap` first"
        raise ValueError(msg)
    return latest.season, latest.round


def _to_round_predictions(
    scored: pl.DataFrame,
    season: int,
    round_number: int,
    model_version: ModelVersion,
) -> RoundPredictions:
    """Turn one round's scored frame into a ranked :class:`RoundPredictions`."""
    ordered = scored.sort(_RANK_COLUMN)
    drivers = [
        DriverPrediction(
            driver_id=row["driver_id"],
            driver_name=row["driver_name"] or row["driver_id"],
            constructor_id=row.get("constructor_id"),
            probability=row[_NORMALIZED_COLUMN],
            probability_calibrated=row[_PROBABILITY_COLUMN],
            probability_raw=row[_RAW_COLUMN],
            rank=row[_RANK_COLUMN],
        )
        for row in ordered.to_dicts()
    ]
    return RoundPredictions(
        season=season,
        round=round_number,
        model_version_id=model_version.id,
        model_name=model_version.name,
        drivers=drivers,
    )
