"""Dataset construction: database data in, a versioned Parquet feature matrix out."""

from f1cp.dataset.builder import DatasetArtifact, DatasetContractError, build_dataset
from f1cp.dataset.loader import load_source_frames
from f1cp.dataset.schema import SchemaViolation, validate_dataset

__all__ = [
    "DatasetArtifact",
    "DatasetContractError",
    "SchemaViolation",
    "build_dataset",
    "load_source_frames",
    "validate_dataset",
]
