"""Typed application settings assembled from ``conf/*.yaml``.

Every subsystem receives a strongly typed configuration object rather than reaching for
environment variables directly. That keeps the dependency direction explicit and makes the
whole application configurable from a single place.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from f1cp.config.loader import CONF_DIR, PROJECT_ROOT, load_yaml_files

__all__ = [
    "AppConfig",
    "DataConfig",
    "DatabaseConfig",
    "ExplainabilityConfig",
    "FeaturesConfig",
    "ModelSpecConfig",
    "PathsConfig",
    "Settings",
    "TrainingConfig",
    "get_settings",
    "load_settings",
]


class _Base(BaseModel):
    """Base model: reject unknown keys so configuration typos fail loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# config.yaml


class AppConfig(_Base):
    """General application metadata."""

    name: str = "f1-champion-predictor"
    environment: str = "local"
    random_seed: int = 42


class PathsConfig(_Base):
    """Filesystem layout. Relative paths are resolved against the project root."""

    data: Path
    raw: Path
    interim: Path
    processed: Path
    cache: Path
    artifacts: Path
    reports: Path

    @model_validator(mode="after")
    def _resolve(self) -> PathsConfig:
        # Frozen models can't go through `self.field = ...`, but pydantic's "after"
        # validator contract expects `self` back (returning a `model_copy` triggers a
        # UserWarning), so fields are updated in place via `object.__setattr__` instead.
        for name, path in self.__dict__.items():
            if not path.is_absolute():
                object.__setattr__(self, name, PROJECT_ROOT / path)
        return self

    def ensure_exist(self) -> None:
        """Create every configured directory if it does not already exist."""
        for path in self.__dict__.values():
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)


class DatabaseConfig(_Base):
    """Database connection settings."""

    url: str
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    pool_pre_ping: bool = True

    @field_validator("url")
    @classmethod
    def _absolutise_sqlite_path(cls, url: str) -> str:
        """Anchor relative SQLite paths to the project root.

        Without this, the database file location depends on the working directory, so the
        API and the training job would silently use different databases.
        """
        prefix = "sqlite:///"
        if not url.startswith(prefix):
            return url
        raw_path = url[len(prefix) :]
        if not raw_path or raw_path.startswith("/") or raw_path == ":memory:":
            return url
        return f"{prefix}{(PROJECT_ROOT / raw_path).resolve()}"

    @property
    def is_sqlite(self) -> bool:
        """Whether the configured backend is SQLite."""
        return self.url.startswith("sqlite")


class MLflowConfig(_Base):
    """Experiment tracking settings.

    Defaults to a SQLite-backed tracking store rather than a plain ``file:`` store: MLflow
    3.x puts the filesystem backend in maintenance mode and refuses to open it without an
    explicit opt-out, so ``file:./data/mlruns`` fails outright on a current install.
    """

    enabled: bool = True
    tracking_uri: str = "sqlite:///./data/mlflow.db"
    experiment_name: str = "f1-championship-predictor"
    registered_model_name: str = "f1-champion-predictor"
    strict: bool = False

    @field_validator("tracking_uri")
    @classmethod
    def _absolutise_relative_uri(cls, uri: str) -> str:
        """Anchor a relative ``file:`` or ``sqlite:///`` tracking URI to the project root.

        Without this, the tracking store location depends on the working directory a
        command happens to be run from.
        """
        for prefix in ("file:", "sqlite:///"):
            if not uri.startswith(prefix):
                continue
            raw_path = uri[len(prefix) :]
            if not raw_path or raw_path.startswith("/") or raw_path == ":memory:":
                return uri
            return f"{prefix}{(PROJECT_ROOT / raw_path).resolve()}"
        return uri


class ApiConfig(_Base):
    """FastAPI service settings."""

    host: str = "0.0.0.0"  # noqa: S104 - container services must bind all interfaces
    port: int = 8000
    title: str = "F1 Championship Predictor API"
    version: str = "0.1.0"
    cors_origins: list[str] = Field(default_factory=list)
    read_only: bool = False


class DashboardConfig(_Base):
    """Streamlit dashboard settings."""

    host: str = "0.0.0.0"  # noqa: S104 - container services must bind all interfaces
    port: int = 8501
    api_base_url: str = "http://localhost:8000"
    request_timeout_seconds: float = 120.0
    page_title: str = "F1 Championship Predictor"


class LoggingConfig(_Base):
    """Logging settings."""

    level: str = "INFO"
    format: Literal["console", "json"] = "console"


# data.yaml


class JolpicaProviderConfig(_Base):
    """Jolpica (Ergast-compatible) REST API settings."""

    base_url: str = "https://api.jolpi.ca/ergast/f1"
    page_limit: int = Field(default=100, le=100, description="Server-side maximum is 100.")
    requests_per_second: float = 3.0
    requests_per_hour: int = 450
    timeout_seconds: float = 30.0
    max_retries: int = 5
    backoff_factor: float = 2.0
    user_agent: str = "f1-champion-predictor/0.1"


class F1dbProviderConfig(_Base):
    """f1db bulk CSV release settings."""

    latest_release_url: str = "https://api.github.com/repos/f1db/f1db/releases/latest"
    asset_name: str = "f1db-csv.zip"
    pinned_tag: str | None = None
    timeout_seconds: float = 120.0


class FastF1ProviderConfig(_Base):
    """FastF1 timing/telemetry settings."""

    cache_dir: Path = Path("data/cache/fastf1")
    min_season: int = 2018
    timeout_seconds: float = 120.0

    @field_validator("cache_dir")
    @classmethod
    def _resolve(cls, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path


class OpenF1ProviderConfig(_Base):
    """OpenF1 REST API settings."""

    base_url: str = "https://api.openf1.org/v1"
    min_season: int = 2023
    requests_per_second: float = 2.0
    requests_per_hour: int = 1500
    timeout_seconds: float = 30.0


class KaggleProviderConfig(_Base):
    """Kaggle dataset settings."""

    dataset: str = "rohanrao/formula-1-world-championship-1950-2020"


class ProvidersConfig(_Base):
    """Container for every provider's settings."""

    jolpica: JolpicaProviderConfig = Field(default_factory=JolpicaProviderConfig)
    f1db: F1dbProviderConfig = Field(default_factory=F1dbProviderConfig)
    fastf1: FastF1ProviderConfig = Field(default_factory=FastF1ProviderConfig)
    openf1: OpenF1ProviderConfig = Field(default_factory=OpenF1ProviderConfig)
    kaggle: KaggleProviderConfig = Field(default_factory=KaggleProviderConfig)


class HttpCacheConfig(_Base):
    """On-disk HTTP response cache settings."""

    enabled: bool = True
    dir: Path = Path("data/cache/http")
    ttl_hours: float = 24.0
    immutable_before_season: int = 2026

    @field_validator("dir")
    @classmethod
    def _resolve(cls, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path


class DataConfig(_Base):
    """Data acquisition configuration."""

    min_season: int = 1981
    max_season: int | None = None
    bootstrap_provider: str = "f1db"
    incremental_provider: str = "jolpica"
    enrichment_providers: list[str] = Field(default_factory=list)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    http_cache: HttpCacheConfig = Field(default_factory=HttpCacheConfig)

    @property
    def resolved_max_season(self) -> int:
        """The last season to include, defaulting to the current calendar year."""
        return self.max_season if self.max_season is not None else dt.date.today().year

    @property
    def season_range(self) -> range:
        """Inclusive season range covered by the modelling dataset."""
        return range(self.min_season, self.resolved_max_season + 1)


# features.yaml


class BlockConfig(_Base):
    """Base configuration shared by every feature block."""

    enabled: bool = True


class StandingsBlockConfig(BlockConfig):
    """Championship standings features."""

    era_normalisation: bool = True


class SeasonContextBlockConfig(BlockConfig):
    """Season progress and remaining-points features."""

    account_for_sprints: bool = True


class FormBlockConfig(BlockConfig):
    """Rolling form and momentum features."""

    windows: list[int] = Field(default_factory=lambda: [3, 5])
    trend_window: int = 5


class QualifyingBlockConfig(BlockConfig):
    """Qualifying and grid-position features."""

    recent_window: int = 3


class ConstructorBlockConfig(BlockConfig):
    """Constructor strength and teammate comparison features."""

    teammate_head_to_head: bool = True


class CircuitBlockConfig(BlockConfig):
    """Historical per-circuit performance features."""

    min_history_races: int = 2
    lookback_seasons: int = 10


class ModernBlockConfig(BlockConfig):
    """FastF1/OpenF1 enrichment features (modern era only)."""

    enabled: bool = False
    min_season: int = 2018


class FeatureBlocksConfig(_Base):
    """Per-block configuration."""

    standings: StandingsBlockConfig = Field(default_factory=StandingsBlockConfig)
    season_context: SeasonContextBlockConfig = Field(default_factory=SeasonContextBlockConfig)
    form: FormBlockConfig = Field(default_factory=FormBlockConfig)
    rates: BlockConfig = Field(default_factory=BlockConfig)
    qualifying: QualifyingBlockConfig = Field(default_factory=QualifyingBlockConfig)
    constructor: ConstructorBlockConfig = Field(default_factory=ConstructorBlockConfig)
    driver: BlockConfig = Field(default_factory=BlockConfig)
    circuit: CircuitBlockConfig = Field(default_factory=CircuitBlockConfig)
    modern: ModernBlockConfig = Field(default_factory=ModernBlockConfig)

    def enabled_names(self) -> list[str]:
        """Names of the enabled blocks, in declaration order."""
        return [name for name, block in self.__dict__.items() if getattr(block, "enabled", False)]


class FeatureFiltersConfig(_Base):
    """Row-level filters applied while building the dataset."""

    min_rounds_in_season: int = 5


class FeaturesConfig(_Base):
    """Feature engineering configuration."""

    key_columns: list[str] = Field(default_factory=lambda: ["season", "round", "driver_id"])
    target_column: str = "champion"
    blocks: FeatureBlocksConfig = Field(default_factory=FeatureBlocksConfig)
    filters: FeatureFiltersConfig = Field(default_factory=FeatureFiltersConfig)
    metadata_columns: list[str] = Field(default_factory=list)


# models.yaml


class ValidationConfig(_Base):
    """Temporal cross-validation settings."""

    strategy: Literal["expanding", "rolling"] = "expanding"
    min_train_seasons: int = 15
    test_horizon_seasons: int = 1
    rolling_window_seasons: int = 15
    calibration_holdout_seasons: int = 3


class CalibrationConfig(_Base):
    """Probability calibration settings.

    ``sigmoid`` (Platt) is the default because isotonic is unsafe on this problem: the
    calibration holdout is a few *seasons*, so only a few independent championship outcomes,
    and isotonic overfits them into 100% title probabilities. ``conf/models.yaml`` sets
    ``sigmoid`` explicitly too, but a default that is only safe while the YAML overrides it
    is a trap for any code path that builds settings without the YAML (docs/AUDIT_v2.md R5).
    """

    enabled: bool = True
    method: Literal["isotonic", "sigmoid"] = "sigmoid"


class NormalisationConfig(_Base):
    """Per-group probability normalisation settings."""

    enabled: bool = True
    group_columns: list[str] = Field(default_factory=lambda: ["season", "round"])


class SelectionConfig(_Base):
    """Model selection settings."""

    primary_metric: str = "group_log_loss"
    direction: Literal["minimize", "maximize"] = "minimize"
    one_standard_error: bool = True
    preference_order: list[str] = Field(default_factory=list)
    tie_breakers: list[str] = Field(default_factory=list)


class EvaluationConfig(_Base):
    """Evaluation reporting settings."""

    phase_bins: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {
            "early": (0.0, 0.34),
            "mid": (0.34, 0.67),
            "late": (0.67, 1.01),
        }
    )
    calibration_bins: int = 10


class ImbalanceConfig(_Base):
    """Class imbalance handling."""

    strategy: Literal["balanced", "none"] = "balanced"


class TrainingConfig(_Base):
    """Everything that governs a training run."""

    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    normalisation: NormalisationConfig = Field(default_factory=NormalisationConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    imbalance: ImbalanceConfig = Field(default_factory=ImbalanceConfig)


class ModelSpecConfig(_Base):
    """Declarative description of one candidate model."""

    enabled: bool = True
    estimator: str
    scale_features: bool = False
    params: dict[str, Any] = Field(default_factory=dict)


class ShapConfig(_Base):
    """SHAP explainability settings."""

    enabled: bool = True
    max_samples: int = 2000
    background_samples: int = 100


class PermutationImportanceConfig(_Base):
    """Permutation importance settings."""

    enabled: bool = True
    n_repeats: int = 10


class ExplainabilityConfig(_Base):
    """Interpretability settings."""

    enabled: bool = True
    shap: ShapConfig = Field(default_factory=ShapConfig)
    permutation_importance: PermutationImportanceConfig = Field(
        default_factory=PermutationImportanceConfig
    )
    top_n_features: int = 25


class Settings(_Base):
    """Root settings object, assembled from every file in ``conf/``."""

    app: AppConfig = Field(default_factory=AppConfig)
    paths: PathsConfig
    database: DatabaseConfig
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    model_zoo: dict[str, ModelSpecConfig] = Field(default_factory=dict)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)

    @property
    def project_root(self) -> Path:
        """Absolute path to the project root."""
        return PROJECT_ROOT

    def enabled_models(self) -> dict[str, ModelSpecConfig]:
        """Candidate models that are switched on.

        Returns:
            Mapping of model name to its specification, preserving declaration order.
        """
        return {name: spec for name, spec in self.model_zoo.items() if spec.enabled}


def load_settings(conf_dir: Path | None = None) -> Settings:
    """Build a :class:`Settings` instance from the YAML configuration directory.

    Args:
        conf_dir: Directory holding the configuration files. Defaults to ``<root>/conf``.

    Returns:
        A fully validated settings object.
    """
    directory = conf_dir or CONF_DIR
    raw = load_yaml_files(
        directory / "config.yaml",
        directory / "data.yaml",
        directory / "features.yaml",
        directory / "models.yaml",
        # Optional, git-ignored escape hatch for machine-specific overrides.
        directory / "local.yaml",
    )
    return Settings.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Returns:
        The cached :class:`Settings` instance.
    """
    return load_settings()
