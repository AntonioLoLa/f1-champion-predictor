"""Thin HTTP client for the FastAPI service.

The dashboard talks to the API exclusively over HTTP -- never to the database or MLflow
directly -- so this is the only module in :mod:`f1cp.dashboard` that knows an endpoint
exists. Every method returns the plain ``dict``/``list[dict]`` JSON the API actually sends,
not the server's pydantic schemas: the dashboard is a separately deployable process that
shares only a wire contract with the API, never its code.
"""

from __future__ import annotations

from typing import Any, cast

import httpx

__all__ = ["ApiError", "DashboardApiClient"]


class ApiError(RuntimeError):
    """Raised when the API responds with an error or cannot be reached."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        """Configure the error.

        Args:
            message: Human-readable description, usually the API's own ``detail``.
            status_code: The HTTP status code, or ``None`` for a connection failure.
        """
        super().__init__(message)
        self.status_code = status_code


class DashboardApiClient:
    """Synchronous HTTP client bound to one API base URL."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float,
        client: httpx.Client | None = None,
    ) -> None:
        """Configure the client.

        Args:
            base_url: The API's base URL, e.g. ``http://localhost:8000``.
            timeout_seconds: Request timeout.
            client: Override the underlying HTTP client entirely. Tests pass Starlette's
                ``TestClient`` here to talk to a FastAPI app in-process -- lifespan
                included, no real socket; production code never sets it.
        """
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_seconds)

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._client.close()

    def health(self) -> dict[str, Any]:
        """Fetch ``/health``."""
        return cast("dict[str, Any]", self._get("/health"))

    def active_model(self) -> dict[str, Any] | None:
        """Fetch ``/model``, or ``None`` if no model has ever been activated."""
        try:
            return cast("dict[str, Any]", self._get("/model"))
        except ApiError as exc:
            if exc.status_code == 404:
                return None
            raise

    def model_versions(self) -> list[dict[str, Any]]:
        """Fetch ``/model/versions``."""
        return cast("list[dict[str, Any]]", self._get("/model/versions"))

    def activate_model(self, model_version_id: int) -> dict[str, Any]:
        """Call ``POST /model/{id}/activate``."""
        return cast("dict[str, Any]", self._post(f"/model/{model_version_id}/activate"))

    def predict(
        self,
        *,
        season: int | None = None,
        round_number: int | None = None,
        model_version_id: int | None = None,
    ) -> dict[str, Any]:
        """Call ``POST /predict``."""
        payload = {"season": season, "round": round_number, "model_version_id": model_version_id}
        return cast("dict[str, Any]", self._post("/predict", json=_drop_none(payload)))

    def predict_season(
        self, season: int, *, model_version_id: int | None = None
    ) -> list[dict[str, Any]]:
        """Fetch ``GET /predict-season/{season}``."""
        params = _drop_none({"model_version_id": model_version_id})
        return cast("list[dict[str, Any]]", self._get(f"/predict-season/{season}", params=params))

    def explain(
        self, season: int, round_number: int, *, model_version_id: int | None = None
    ) -> dict[str, Any]:
        """Fetch ``GET /explain/{season}/{round}``."""
        params = _drop_none({"model_version_id": model_version_id})
        return cast("dict[str, Any]", self._get(f"/explain/{season}/{round_number}", params=params))

    def standings(self, season: int, *, round_number: int | None = None) -> dict[str, Any]:
        """Fetch ``GET /seasons/{season}/standings``."""
        params = _drop_none({"round": round_number})
        return cast("dict[str, Any]", self._get(f"/seasons/{season}/standings", params=params))

    def seasons(self) -> dict[str, Any]:
        """Fetch ``GET /seasons``."""
        return cast("dict[str, Any]", self._get("/seasons"))

    def season_races(self, season: int) -> dict[str, Any]:
        """Fetch ``GET /seasons/{season}/races``."""
        return cast("dict[str, Any]", self._get(f"/seasons/{season}/races"))

    def race_results(self, season: int, round_number: int) -> list[dict[str, Any]]:
        """Fetch ``GET /seasons/{season}/races/{round}/results``."""
        return cast(
            "list[dict[str, Any]]", self._get(f"/seasons/{season}/races/{round_number}/results")
        )

    def pit_stops(self, season: int, round_number: int) -> list[dict[str, Any]]:
        """Fetch ``GET /seasons/{season}/races/{round}/pit-stops``."""
        return cast(
            "list[dict[str, Any]]",
            self._get(f"/seasons/{season}/races/{round_number}/pit-stops"),
        )

    def constructor_standings(
        self, season: int, *, round_number: int | None = None
    ) -> list[dict[str, Any]]:
        """Fetch ``GET /seasons/{season}/constructor-standings``."""
        params = _drop_none({"round": round_number})
        return cast(
            "list[dict[str, Any]]",
            self._get(f"/seasons/{season}/constructor-standings", params=params),
        )

    def title_contention(self, season: int, *, round_number: int | None = None) -> dict[str, Any]:
        """Fetch ``GET /seasons/{season}/title-contention``."""
        params = _drop_none({"round": round_number})
        return cast(
            "dict[str, Any]", self._get(f"/seasons/{season}/title-contention", params=params)
        )

    def train(self, *, dataset_version: str | None = None) -> dict[str, Any]:
        """Call ``POST /train``."""
        payload = _drop_none({"dataset_version": dataset_version})
        return cast("dict[str, Any]", self._post("/train", json=payload))

    def train_status(self, run_uuid: str) -> dict[str, Any]:
        """Fetch ``GET /train/{run_uuid}``."""
        return cast("dict[str, Any]", self._get(f"/train/{run_uuid}"))

    def update_data(
        self,
        *,
        bootstrap: bool = False,
        seasons: list[int] | None = None,
        refresh: bool = False,
        min_season: int | None = None,
        background: bool = False,
    ) -> dict[str, Any]:
        """Call ``POST /update-data``."""
        payload = {
            "bootstrap": bootstrap,
            "seasons": seasons,
            "refresh": refresh,
            "min_season": min_season,
            "background": background,
        }
        return cast("dict[str, Any]", self._post("/update-data", json=_drop_none(payload)))

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        try:
            response = self._client.get(path, params=params)
        except httpx.HTTPError as exc:
            raise ApiError(f"Could not reach the API: {exc}") from exc
        return _unwrap(response)

    def _post(self, path: str, *, json: dict[str, Any] | None = None) -> Any:
        try:
            response = self._client.post(path, json=json)
        except httpx.HTTPError as exc:
            raise ApiError(f"Could not reach the API: {exc}") from exc
        return _unwrap(response)


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove ``None`` values so optional query/body fields are omitted, not sent as null."""
    return {key: value for key, value in payload.items() if value is not None}


def _unwrap(response: httpx.Response) -> Any:
    """Return the parsed JSON body, or raise :class:`ApiError` with the API's own detail."""
    if response.is_success:
        return response.json()
    raise ApiError(_error_detail(response), status_code=response.status_code)


def _error_detail(response: httpx.Response) -> str:
    """Extract FastAPI's ``{"detail": ...}`` body, falling back to the raw response."""
    try:
        body = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])
    return str(body)
