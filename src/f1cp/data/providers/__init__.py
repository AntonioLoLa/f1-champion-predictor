"""Data providers and the abstraction that makes them interchangeable."""

from f1cp.data.providers.base import (
    BulkHistoryProvider,
    Capability,
    DataBundle,
    DataProvider,
    EnrichmentProvider,
    ProviderError,
    SeasonProvider,
)
from f1cp.data.providers.f1db import F1dbProvider
from f1cp.data.providers.fastf1_provider import FastF1Provider
from f1cp.data.providers.jolpica import JolpicaProvider
from f1cp.data.providers.kaggle import KaggleProvider
from f1cp.data.providers.openf1 import OpenF1Provider
from f1cp.data.providers.registry import ProviderRegistry

__all__ = [
    "BulkHistoryProvider",
    "Capability",
    "DataBundle",
    "DataProvider",
    "EnrichmentProvider",
    "F1dbProvider",
    "FastF1Provider",
    "JolpicaProvider",
    "KaggleProvider",
    "OpenF1Provider",
    "ProviderError",
    "ProviderRegistry",
    "SeasonProvider",
]
