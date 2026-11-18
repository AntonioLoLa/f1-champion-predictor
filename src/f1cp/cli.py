"""Command-line interface.

The scripts under ``scripts/`` are thin wrappers over these commands, so there is exactly
one implementation of each pipeline stage.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from f1cp import __version__
from f1cp.container import AppContext
from f1cp.domain.enums import RunStatus

app = typer.Typer(
    name="f1cp",
    help="F1 Championship Predictor: data, features, training and serving.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Data acquisition commands.", no_args_is_help=True)
app.add_typer(data_app, name="data")
dataset_app = typer.Typer(help="Dataset build commands.", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")
train_app = typer.Typer(help="Model training commands.", no_args_is_help=True)
app.add_typer(train_app, name="train")
evaluate_app = typer.Typer(help="Model evaluation commands.", no_args_is_help=True)
app.add_typer(evaluate_app, name="evaluate")

console = Console()

__all__ = ["app"]


@app.callback()
def main() -> None:
    """F1 Championship Predictor command line."""


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"f1-champion-predictor [bold cyan]{__version__}[/]")


@app.command()
def info() -> None:
    """Show the effective configuration and what is currently stored."""
    with AppContext.create() as context:
        settings = context.settings

        table = Table(title="Configuration", show_header=False, box=None)
        table.add_row("environment", settings.app.environment)
        table.add_row("database", context.database._sanitised_url())  # noqa: SLF001
        table.add_row("mlflow", settings.mlflow.tracking_uri)
        table.add_row("seasons", f"{settings.data.min_season}-{settings.data.resolved_max_season}")
        table.add_row("bootstrap provider", settings.data.bootstrap_provider)
        table.add_row("incremental provider", settings.data.incremental_provider)
        table.add_row("enrichment providers", ", ".join(settings.data.enrichment_providers) or "-")
        table.add_row("feature blocks", ", ".join(settings.features.blocks.enabled_names()))
        table.add_row("models", ", ".join(settings.enabled_models()))
        console.print(table)

        with context.database.session_scope() as uow:
            latest = uow.races.latest_completed()
            stored = Table(title="Stored data", show_header=False, box=None)
            stored.add_row("seasons", str(len(uow.races.seasons())))
            stored.add_row("races", str(uow.races.count()))
            stored.add_row("race results", str(uow.results.count()))
            stored.add_row("drivers", str(uow.drivers.count()))
            stored.add_row(
                "latest round",
                f"{latest.season} R{latest.round} - {latest.name}" if latest else "none",
            )
            console.print(stored)


@data_app.command("update")
def data_update(
    bootstrap: Annotated[
        bool,
        typer.Option("--bootstrap", help="Seed the full history from the bulk provider first."),
    ] = False,
    season: Annotated[
        list[int] | None,
        typer.Option("--season", help="Season to update. Repeatable. Defaults to the current one."),
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option(
            "--refresh", help="Re-fetch the given seasons in full, ignoring what is stored."
        ),
    ] = False,
    min_season: Annotated[
        int | None,
        typer.Option("--min-season", help="Earliest season to bootstrap."),
    ] = None,
) -> None:
    """Acquire data: bootstrap the history and/or fetch new rounds.

    A first run needs ``--bootstrap``; afterwards the plain command is enough, and costs
    roughly seven API requests per new round.
    """
    with AppContext.create(ensure_schema=True) as context:
        ingestor = context.ingestor()
        reports = []

        if bootstrap:
            console.print("[bold]Bootstrapping full history[/]")
            reports.append(ingestor.bootstrap(min_season=min_season))

        if refresh:
            if not season:
                console.print("[red]--refresh requires at least one --season[/]")
                raise typer.Exit(code=2)
            for year in season:
                console.print(f"[bold]Refreshing season {year}[/]")
                reports.append(ingestor.refresh_season(year))
        elif not bootstrap or season:
            console.print("[bold]Fetching missing rounds[/]")
            reports.append(ingestor.incremental(seasons=season))

        _print_reports(reports)
        if any(report.status is RunStatus.FAILED for report in reports):
            raise typer.Exit(code=1)


@data_app.command("enrich")
def data_enrich(
    provider: Annotated[str, typer.Argument(help="Enrichment provider: fastf1 or openf1.")],
    season: Annotated[
        list[int] | None, typer.Option("--season", help="Season to enrich. Repeatable.")
    ] = None,
) -> None:
    """Fetch session-level enrichment metrics and store them as Parquet."""
    with AppContext.create() as context:
        frame = context.ingestor().enrich(provider, seasons=season)
        if not frame.height:
            console.print("[yellow]No enrichment data returned[/]")
            return

        destination = context.settings.paths.interim / f"enrichment_{provider}.parquet"
        frame.write_parquet(destination)
        console.print(f"Wrote [bold]{frame.height}[/] rows to {destination}")


@dataset_app.command("build")
def dataset_build() -> None:
    """Build the model-ready feature matrix from curated data and register its version."""
    from f1cp.dataset import DatasetContractError, build_dataset
    from f1cp.db.models import DatasetVersion

    with AppContext.create() as context:
        with context.database.session_scope() as uow:
            try:
                artifact = build_dataset(uow, context.settings)
            except DatasetContractError as exc:
                console.print(f"[red]{exc}[/]")
                raise typer.Exit(code=1) from exc

            existing = uow.datasets.get_by_version(artifact.version)
            if existing is not None:
                # Rebuilding the same day with unchanged configuration produces the same
                # content-addressed version string; replace rather than fail, since the
                # underlying data (a re-run of `data update`) may have moved on.
                uow.session.delete(existing)
                uow.session.flush()
            uow.datasets.add(
                DatasetVersion(
                    version=artifact.version,
                    path=artifact.path,
                    n_rows=artifact.n_rows,
                    n_features=artifact.n_features,
                    min_season=artifact.min_season,
                    max_season=artifact.max_season,
                    feature_names=artifact.feature_names,
                    config_hash=artifact.config_hash,
                )
            )

        table = Table(title="Dataset built", show_header=False, box=None)
        table.add_row("version", artifact.version)
        table.add_row("rows", str(artifact.n_rows))
        table.add_row("features", str(artifact.n_features))
        table.add_row("seasons", f"{artifact.min_season}-{artifact.max_season}")
        table.add_row("path", artifact.path)
        console.print(table)


@train_app.command("run")
def train_run(
    dataset_version: Annotated[
        str | None,
        typer.Option(
            "--dataset-version", help="Dataset version to train on. Defaults to the latest build."
        ),
    ] = None,
) -> None:
    """Fit, calibrate, score and select a model from the current model zoo."""
    with AppContext.create(ensure_schema=True) as context:
        result = context.trainer().run(dataset_version=dataset_version)

    table = Table(title="Training run", show_header=True, box=None)
    table.add_column("model")
    table.add_column("group_log_loss", justify="right")
    table.add_column("brier_score", justify="right")
    table.add_column("expected_calibration_error", justify="right")
    table.add_column("top1_hit_rate", justify="right")
    table.add_column("winner", justify="center")
    for candidate in result.candidates:
        table.add_row(
            candidate.name,
            _format_metric(candidate.metrics.get("group_log_loss")),
            _format_metric(candidate.metrics.get("brier_score")),
            _format_metric(candidate.metrics.get("expected_calibration_error")),
            _format_metric(candidate.metrics.get("top1_hit_rate")),
            "*" if candidate.name == result.best_model_name else "",
        )
    console.print(table)
    console.print(
        f"Selected [bold cyan]{result.best_model_name}[/] "
        f"(model_version_id={result.best_model_version_id}, run_uuid={result.run_uuid})"
    )


@train_app.command("due")
def train_due() -> None:
    """Print ``true`` if a season has closed since the last successful training run.

    Retraining mid-season fits the same labelled seasons as the previous run, so scheduled
    automation asks this first (see :mod:`f1cp.training.cadence`). The answer goes to
    stdout rather than the exit code, so a crash can never read as "not due".
    """
    from f1cp.training.cadence import retraining_due

    with AppContext.create(ensure_schema=True) as context, context.database.session_scope() as uow:
        typer.echo("true" if retraining_due(uow) else "false")


@evaluate_app.command("run")
def evaluate_run(
    model_version_id: Annotated[
        int | None,
        typer.Option(
            "--model-version-id", help="Model version to evaluate. Defaults to the active one."
        ),
    ] = None,
) -> None:
    """Score a model's out-of-fold predictions, run explainability, and write a report."""
    from f1cp.evaluation import build_report

    with AppContext.create(ensure_schema=True) as context, context.database.session_scope() as uow:
        try:
            report = build_report(uow, context.settings, model_version_id=model_version_id)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc

    table = Table(title="Evaluation report", show_header=False, box=None)
    table.add_row("model version", str(report.model_version_id))
    table.add_row("report directory", report.report_dir)
    for metric_name in ("accuracy", "roc_auc", "pr_auc", "log_loss", "top1_hit_rate"):
        table.add_row(metric_name, _format_metric(report.pooled_metrics.get(metric_name)))
    console.print(table)


@app.command("db-upgrade")
def db_upgrade(
    revision: Annotated[str, typer.Argument(help="Target revision.")] = "head",
) -> None:
    """Apply Alembic migrations."""
    from alembic import command
    from alembic.config import Config

    from f1cp.config import PROJECT_ROOT

    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    command.upgrade(config, revision)
    console.print(f"Database upgraded to [bold cyan]{revision}[/]")


def _format_metric(value: float | None) -> str:
    """Render an optional metric value for a table cell."""
    return "-" if value is None else f"{value:.4f}"


def _print_reports(reports: list) -> None:  # type: ignore[type-arg]
    """Render ingestion reports as a table."""
    for report in reports:
        colour = "green" if report.status is RunStatus.SUCCEEDED else "red"
        console.print(
            f"  [{colour}]{report.status}[/] {report.mode} via {report.provider}"
            f" - {report.total_rows} rows"
        )
        if report.error:
            console.print(f"  [red]{report.error}[/]")
        if report.rows_written:
            table = Table(show_header=True, box=None, pad_edge=False)
            table.add_column("table")
            table.add_column("rows", justify="right")
            for name, count in sorted(report.rows_written.items()):
                table.add_row(str(name), str(count))
            console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
