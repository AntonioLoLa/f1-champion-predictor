"""Evaluation report generation.

Builds a full report for one trained model: the classification/ranking metrics
``training.metrics`` doesn't already compute (pooled, per season phase, and per season --
new), Plotly figures for all of them, and SHAP + permutation importance against the
deployed model's pre-calibration pipeline. Runs entirely from what training already
persisted -- the out-of-fold predictions parquet and the logged base pipeline -- so it can
be re-run independently, any time, against any past model version, without re-fitting
anything.

PNG export needs a working Chromium (via ``kaleido``/``plotly_get_chrome``), which is not
guaranteed on every machine this ever runs on; a failure there is caught and logged rather
than failing the whole report, since the JSON figure (needed by anything that renders it
interactively) never depends on a browser at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sqlalchemy import delete, or_

from f1cp.db.models import Evaluation
from f1cp.evaluation.metrics import calibration_curve_points, compute_full_metrics
from f1cp.evaluation.plots import (
    calibration_curve_figure,
    confusion_matrix_figure,
    metric_by_phase_figure,
    metric_by_season_figure,
    pr_curve_figure,
    roc_curve_figure,
)
from f1cp.explain.permutation import explain as explain_permutation
from f1cp.explain.plots import permutation_importance_figure, shap_importance_figure
from f1cp.explain.shap_explainer import explain as explain_shap
from f1cp.logging_config import get_logger
from f1cp.training.metrics import compute_metrics as compute_selection_metrics

if TYPE_CHECKING:
    from pathlib import Path

    import plotly.graph_objects as go

    from f1cp.config.settings import Settings
    from f1cp.db.models import ModelVersion
    from f1cp.db.uow import UnitOfWork

__all__ = ["EvaluationReport", "build_report"]

logger = get_logger(__name__)

_PROBABILITY_COLUMN = "probability"
_NORMALIZED_COLUMN = "probability_normalized"

# Metric names this module (never training) writes -- used to make re-running a report
# idempotent without touching training's own pooled/phase/fold rows.
_NEW_METRIC_NAMES = frozenset(
    {
        "accuracy",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "roc_auc",
        "pr_auc",
        "log_loss",
    }
)


@dataclass(slots=True)
class EvaluationReport:
    """Outcome of one :func:`build_report` call.

    Attributes:
        model_version_id: The model version this report was built for.
        report_dir: Directory the figures, metrics and ``index.html`` were written to.
        pooled_metrics: Every pooled metric (training's selection metrics plus this
            module's fuller suite), for a quick summary.
        phase_metrics: Same, per season phase.
        season_metrics: Same, per season -- a granularity training never computes.
    """

    model_version_id: int
    report_dir: str
    pooled_metrics: dict[str, float | None]
    phase_metrics: dict[str, dict[str, float | None]]
    season_metrics: dict[int, dict[str, float | None]]


def build_report(
    uow: UnitOfWork,
    settings: Settings,
    *,
    model_version_id: int | None = None,
) -> EvaluationReport:
    """Score a trained model's out-of-fold predictions and render a full report.

    Args:
        uow: An open unit of work to read/write model metadata through.
        settings: Application settings.
        model_version_id: Which model version to evaluate. Defaults to the active one.

    Returns:
        A summary of every metric computed and where the report was written.

    Raises:
        ValueError: If no matching model version exists, or it predates this feature and
            has no recorded out-of-fold predictions.
    """
    model_version = _resolve_model_version(uow, model_version_id)
    if not model_version.oof_predictions_path:
        msg = (
            f"Model version {model_version.id} has no recorded out-of-fold predictions "
            "(it was trained before this existed); re-run `f1cp train run`."
        )
        raise ValueError(msg)

    target_column = settings.features.target_column
    group_columns = settings.training.normalisation.group_columns
    calibration_bins = settings.training.evaluation.calibration_bins
    oof_frame = pl.read_parquet(model_version.oof_predictions_path)

    def _full_metrics(frame: pl.DataFrame) -> dict[str, float | None]:
        return compute_full_metrics(
            frame,
            calibrated_column=_PROBABILITY_COLUMN,
            normalized_column=_NORMALIZED_COLUMN,
            target_column=target_column,
            group_columns=group_columns,
        )

    pooled_new = _full_metrics(oof_frame)
    pooled_metrics = {
        **_existing_metrics(uow, model_version.id, phase=None),
        **pooled_new,
    }

    phase_metrics_new: dict[str, dict[str, float | None]] = {}
    phase_metrics: dict[str, dict[str, float | None]] = {}
    phases = sorted(p for p in oof_frame["phase"].unique().to_list() if p is not None)
    for phase_name in phases:
        new = _full_metrics(oof_frame.filter(pl.col("phase") == phase_name))
        phase_metrics_new[phase_name] = new
        phase_metrics[phase_name] = {
            **_existing_metrics(uow, model_version.id, phase=phase_name),
            **new,
        }

    season_metrics: dict[int, dict[str, float | None]] = {}
    for season in sorted(oof_frame["season"].unique().to_list()):
        subset = oof_frame.filter(pl.col("season") == season)
        selection = compute_selection_metrics(
            subset,
            calibrated_column=_PROBABILITY_COLUMN,
            normalized_column=_NORMALIZED_COLUMN,
            target_column=target_column,
            group_columns=group_columns,
            calibration_bins=calibration_bins,
        )
        season_metrics[season] = {**selection, **_full_metrics(subset)}

    report_dir = settings.paths.reports / str(model_version.id)
    report_dir.mkdir(parents=True, exist_ok=True)

    figures: dict[str, go.Figure] = {
        "calibration_curve": calibration_curve_figure(
            calibration_curve_points(
                oof_frame,
                calibrated_column=_PROBABILITY_COLUMN,
                target_column=target_column,
                n_bins=calibration_bins,
            )
        ),
        "confusion_matrix": confusion_matrix_figure(pooled_metrics),
        "roc_curve": roc_curve_figure(
            oof_frame, calibrated_column=_PROBABILITY_COLUMN, target_column=target_column
        ),
        "pr_curve": pr_curve_figure(
            oof_frame, calibrated_column=_PROBABILITY_COLUMN, target_column=target_column
        ),
        "top1_hit_rate_by_phase": metric_by_phase_figure(
            phase_metrics, metric_name="top1_hit_rate"
        ),
        "top1_hit_rate_by_season": metric_by_season_figure(
            season_metrics, metric_name="top1_hit_rate"
        ),
    }

    _add_explainability_figures(settings, model_version, figures)

    for name, figure in figures.items():
        _write_figure(figure, report_dir / name)

    _delete_previous_report_rows(uow, model_version.id)
    uow.evaluations.add_many(
        _evaluation_rows(model_version.id, pooled_new, phase_metrics_new, season_metrics)
    )
    _write_metrics_json(report_dir, pooled_metrics, phase_metrics, season_metrics)
    _write_index_html(report_dir, model_version, pooled_metrics, figures)

    return EvaluationReport(
        model_version_id=model_version.id,
        report_dir=str(report_dir),
        pooled_metrics=pooled_metrics,
        phase_metrics=phase_metrics,
        season_metrics=season_metrics,
    )


def _resolve_model_version(uow: UnitOfWork, model_version_id: int | None) -> ModelVersion:
    """Look up the model version to evaluate, defaulting to the active one."""
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


def _existing_metrics(
    uow: UnitOfWork, model_version_id: int, *, phase: str | None
) -> dict[str, float | None]:
    """Metrics training already persisted at this granularity (pooled or per-phase).

    ``EvaluationRepository.for_model``'s ``phase=None`` means "every phase, unfiltered" --
    exactly wrong for a caller that wants the *pooled* (``phase IS NULL``) rows
    specifically, so ``phase``, ``fold`` and ``season`` are all re-checked client-side
    rather than trusting the query alone; otherwise a season- or another phase's row
    sharing the same metric name silently wins whichever one the query happens to return
    last, rather than the one this granularity actually asked for.
    """
    rows = uow.evaluations.for_model(model_version_id, phase=phase)
    return {
        row.metric_name: row.metric_value
        for row in rows
        if row.fold is None and row.phase == phase and row.season is None
    }


def _add_explainability_figures(
    settings: Settings,
    model_version: ModelVersion,
    figures: dict[str, go.Figure],
) -> None:
    """Run SHAP / permutation importance on the deployed model and add their figures."""
    explainability = settings.explainability
    if not explainability.enabled or not model_version.mlflow_base_model_uri:
        return

    dataset_version = model_version.training_run.dataset_version
    if dataset_version is None:
        return

    import mlflow
    import mlflow.sklearn  # local import: only needed on the explainability path

    # `evaluate` never constructs an `ExperimentTracker` (nothing here logs a run), so
    # nothing else has pointed MLflow at the configured store; without this, loading falls
    # back to MLflow's own default tracking URI and silently fails to find the model.
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

    full_frame = pl.read_parquet(dataset_version.path)
    feature_names = list(model_version.feature_names or dataset_version.feature_names)
    x_full = full_frame.select(feature_names).cast(pl.Float64).to_numpy()
    base_pipeline = mlflow.sklearn.load_model(model_version.mlflow_base_model_uri)

    if explainability.shap.enabled:
        shap_result = explain_shap(
            base_pipeline,
            model_version.algorithm,
            x_full,
            feature_names,
            explainability.shap,
            random_seed=settings.app.random_seed,
        )
        figures["shap_importance"] = shap_importance_figure(
            shap_result, top_n=explainability.top_n_features
        )

    if explainability.permutation_importance.enabled:
        sample_size = min(explainability.shap.max_samples, full_frame.height)
        rng = np.random.default_rng(settings.app.random_seed)
        sample_idx = rng.choice(full_frame.height, size=sample_size, replace=False)
        y_full = full_frame[settings.features.target_column].to_numpy()
        permutation_result = explain_permutation(
            base_pipeline,
            x_full[sample_idx],
            y_full[sample_idx],
            feature_names,
            explainability.permutation_importance,
            random_seed=settings.app.random_seed,
        )
        figures["permutation_importance"] = permutation_importance_figure(
            permutation_result, top_n=explainability.top_n_features
        )


def _write_figure(figure: go.Figure, path_without_ext: Path) -> None:
    """Write a figure as JSON (always) and PNG (best-effort)."""
    figure.write_json(f"{path_without_ext}.json")
    try:
        figure.write_image(f"{path_without_ext}.png")
    except Exception:
        logger.warning(
            "Could not render PNG for '%s' (kaleido/Chrome unavailable?); JSON was still written",
            path_without_ext.name,
        )


def _delete_previous_report_rows(uow: UnitOfWork, model_version_id: int) -> None:
    """Delete this module's own rows from a prior report, so re-running one is idempotent.

    Scoped to exactly what :func:`_evaluation_rows` is about to (re-)insert: rows with one
    of this module's metric names (the pooled/phase granularity) or any season-level row
    at all (a granularity training never writes, regardless of metric name). Training's
    own pooled/phase/fold rows are untouched.
    """
    uow.session.execute(
        delete(Evaluation).where(
            Evaluation.model_version_id == model_version_id,
            or_(Evaluation.metric_name.in_(_NEW_METRIC_NAMES), Evaluation.season.is_not(None)),
        )
    )


def _evaluation_rows(
    model_version_id: int,
    pooled_new: dict[str, float | None],
    phase_metrics_new: dict[str, dict[str, float | None]],
    season_metrics: dict[int, dict[str, float | None]],
) -> list[dict[str, object]]:
    """Flatten the metrics this module adds into long-format rows.

    Only what's new here: pooled/phase rows already computed by training are read back,
    not re-inserted; every season-level row is new, since training never persists that
    granularity.
    """
    rows: list[dict[str, object]] = []

    def _add(*, season: int | None, phase: str | None, metrics: dict[str, float | None]) -> None:
        for metric_name, value in metrics.items():
            if value is None:
                continue
            rows.append(
                {
                    "model_version_id": model_version_id,
                    "split": "validation",
                    "fold": None,
                    "season": season,
                    "phase": phase,
                    "metric_name": metric_name,
                    "metric_value": value,
                }
            )

    _add(season=None, phase=None, metrics=pooled_new)
    for phase_name, metrics in phase_metrics_new.items():
        _add(season=None, phase=phase_name, metrics=metrics)
    for season, metrics in season_metrics.items():
        _add(season=season, phase=None, metrics=metrics)
    return rows


def _write_metrics_json(
    report_dir: Path,
    pooled_metrics: dict[str, float | None],
    phase_metrics: dict[str, dict[str, float | None]],
    season_metrics: dict[int, dict[str, float | None]],
) -> None:
    payload = {"pooled": pooled_metrics, "phase": phase_metrics, "season": season_metrics}
    (report_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str)
    )


def _write_index_html(
    report_dir: Path,
    model_version: ModelVersion,
    pooled_metrics: dict[str, float | None],
    figures: dict[str, go.Figure],
) -> None:
    metric_rows = "".join(
        f"<tr><td>{name}</td><td>{'-' if value is None else f'{value:.4f}'}</td></tr>"
        for name, value in sorted(pooled_metrics.items())
    )
    figure_divs = "".join(
        figure.to_html(full_html=False, include_plotlyjs=False, div_id=name)
        for name, figure in figures.items()
    )
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Evaluation report: {model_version.name} (v{model_version.id})</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<h1>Evaluation report: {model_version.name}</h1>
<p>Model version {model_version.id} &mdash; algorithm {model_version.algorithm}</p>
<h2>Pooled metrics</h2>
<table border="1" cellpadding="4">
<tr><th>Metric</th><th>Value</th></tr>
{metric_rows}
</table>
<h2>Figures</h2>
{figure_divs}
</body>
</html>
"""
    (report_dir / "index.html").write_text(html)
