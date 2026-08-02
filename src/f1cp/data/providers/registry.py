"""Provider factory.

Resolves provider names from configuration into constructed instances. This is the seam
that makes the sources swappable: changing ``data.bootstrap_provider`` from ``f1db`` to
``kaggle`` in YAML is the only edit required to change where the history comes from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from f1cp.data.http import ResponseCache
from f1cp.data.providers.base import (
    BulkHistoryProvider,
    Capability,
    EnrichmentProvider,
    ProviderError,
    SeasonProvider,
)
from f1cp.data.providers.f1db import F1dbProvider
from f1cp.data.providers.fastf1_provider import FastF1Provider
from f1cp.data.providers.jolpica import JolpicaProvider
from f1cp.data.providers.kaggle import KaggleProvider
from f1cp.data.providers.openf1 import OpenF1Provider
from f1cp.domain.enums import ProviderName

if TYPE_CHECKING:
    from collections.abc import Mapping

    from f1cp.config.settings import Settings

__all__ = ["ProviderRegistry"]


class ProviderRegistry:
    """Builds providers from settings, sharing one HTTP cache between them."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        cache_config = settings.data.http_cache
        self._cache = ResponseCache(
            cache_config.dir,
            ttl_hours=cache_config.ttl_hours,
            enabled=cache_config.enabled,
        )

    @property
    def cache(self) -> ResponseCache:
        """The shared on-disk response cache."""
        return self._cache

    def bulk(self, name: str | None = None) -> BulkHistoryProvider:
        """Build a bulk-history provider.

        Args:
            name: Provider name, defaulting to ``data.bootstrap_provider``.

        Returns:
            A provider capable of downloading the full history.

        Raises:
            ProviderError: If the name is unknown or the provider cannot bulk-load.
        """
        resolved = name or self._settings.data.bootstrap_provider
        data = self._settings.data

        match resolved:
            case ProviderName.F1DB:
                return F1dbProvider(data.providers.f1db, download_dir=self._settings.paths.raw)
            case ProviderName.KAGGLE:
                return KaggleProvider(data.providers.kaggle, download_dir=self._settings.paths.raw)
            case _:
                msg = (
                    f"Provider {resolved!r} cannot bulk-load history. "
                    f"Supported: {ProviderName.F1DB}, {ProviderName.KAGGLE}"
                )
                raise ProviderError(msg)

    def season(self, name: str | None = None) -> SeasonProvider:
        """Build a per-season provider for incremental updates.

        Args:
            name: Provider name, defaulting to ``data.incremental_provider``.

        Returns:
            A provider capable of fetching individual seasons.

        Raises:
            ProviderError: If the name is unknown or the provider cannot fetch seasons.
        """
        resolved = name or self._settings.data.incremental_provider

        if resolved == ProviderName.JOLPICA:
            return JolpicaProvider(
                self._settings.data.providers.jolpica,
                cache=self._cache,
                immutable_before_season=self._settings.data.http_cache.immutable_before_season,
            )

        msg = f"Provider {resolved!r} cannot fetch individual seasons. Supported: jolpica"
        raise ProviderError(msg)

    def enrichment(
        self,
        name: str,
        *,
        driver_lookup: Mapping[str, str] | None = None,
    ) -> EnrichmentProvider:
        """Build an enrichment provider.

        Args:
            name: Provider name.
            driver_lookup: Normalised full name to canonical driver id, used to reconcile
                the provider's own driver identifiers.

        Returns:
            An enrichment provider.

        Raises:
            ProviderError: If the name is unknown or the provider offers no enrichment.
        """
        providers = self._settings.data.providers

        match name:
            case ProviderName.FASTF1:
                return FastF1Provider(providers.fastf1, driver_lookup=driver_lookup)
            case ProviderName.OPENF1:
                return OpenF1Provider(
                    providers.openf1, cache=self._cache, driver_lookup=driver_lookup
                )
            case _:
                msg = (
                    f"Provider {name!r} offers no enrichment. "
                    f"Supported: {ProviderName.FASTF1}, {ProviderName.OPENF1}"
                )
                raise ProviderError(msg)

    def describe(self) -> dict[str, list[str]]:
        """Summarise which provider fills which role, for logging and the API.

        Returns:
            Mapping of capability to the configured provider names.
        """
        return {
            Capability.BULK_HISTORY: [self._settings.data.bootstrap_provider],
            Capability.SEASON_FETCH: [self._settings.data.incremental_provider],
            Capability.ENRICHMENT: list(self._settings.data.enrichment_providers),
        }
