"""YAML configuration loading with environment-variable expansion.

The loader keeps configuration declarative (YAML) while allowing every value to be
overridden from the environment, which is what makes the same files usable from a local
shell, a Docker Compose service and a CI job without modification.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = ["CONF_DIR", "PROJECT_ROOT", "deep_merge", "expand_env_vars", "load_yaml_files"]

# Matches ``${NAME}`` and ``${NAME:-default}`` (shell / docker-compose syntax).
_ENV_PATTERN = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}",
)

_ROOT_MARKERS = ("pyproject.toml", "conf")


@lru_cache(maxsize=1)
def _discover_project_root() -> Path:
    """Locate the project root.

    Resolution order: the ``F1CP_ROOT`` environment variable, then the nearest ancestor of
    this file containing a root marker, then the current working directory. The fallbacks
    matter because the package is installed non-editable inside the Docker images.

    Returns:
        Absolute path to the project root.
    """
    if env_root := os.environ.get("F1CP_ROOT"):
        return Path(env_root).resolve()

    for candidate in Path(__file__).resolve().parents:
        if all((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate

    return Path.cwd().resolve()


PROJECT_ROOT: Path = _discover_project_root()
CONF_DIR: Path = PROJECT_ROOT / "conf"


def expand_env_vars(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` placeholders.

    Args:
        value: Any YAML-derived structure (mapping, sequence, scalar).

    Returns:
        The same structure with string placeholders resolved against the environment.

    Raises:
        KeyError: If a placeholder has no default and the variable is unset.
    """
    if isinstance(value, Mapping):
        return {key: expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return _expand_string(value)
    return value


def _expand_string(raw: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        name = match.group("name")
        default = match.group("default")
        if (resolved := os.environ.get(name)) is not None:
            return resolved
        if default is not None:
            return default
        msg = f"Environment variable {name!r} is referenced in configuration but not set"
        raise KeyError(msg)

    return _ENV_PATTERN.sub(_replace, raw)


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either argument.

    Args:
        base: The lower-precedence mapping.
        override: The higher-precedence mapping.

    Returns:
        A new merged dictionary.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def load_yaml_files(*paths: Path) -> dict[str, Any]:
    """Load and merge YAML files, expanding environment placeholders.

    Later files take precedence over earlier ones.

    Args:
        *paths: YAML files to load. Missing files are skipped so that optional local
            overrides (``conf/local.yaml``) do not need to exist.

    Returns:
        The merged configuration mapping.
    """
    import yaml  # imported lazily to keep module import cost low

    merged: dict[str, Any] = {}
    for path in paths:
        if not path.exists():
            continue
        content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(content, dict):
            msg = f"Configuration file {path} must contain a mapping at the top level"
            raise TypeError(msg)
        merged = deep_merge(merged, content)
    return expand_env_vars(merged)  # type: ignore[no-any-return]
