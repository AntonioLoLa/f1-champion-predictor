"""Rate-limited, cached JSON HTTP client.

Jolpica allows 4 requests per second and 500 per hour, and caps page size at 100 rows.
Reconstructing the championship from 1950 means tens of thousands of rows, so respecting
those limits is a correctness requirement, not politeness: exceeding them returns 429s and
the backfill silently loses data.

Three mechanisms handle this:

* a dual token bucket enforcing the per-second *and* per-hour budgets;
* an on-disk response cache, with responses for completed seasons treated as immutable and
  never re-fetched;
* bounded retries with exponential backoff that honour the ``Retry-After`` header.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx

from f1cp.logging_config import get_logger

__all__ = ["HttpError", "JsonHttpClient", "RateLimit", "RateLimiter", "ResponseCache"]

logger = get_logger(__name__)

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_SECONDS_PER_HOUR = 3600.0


class HttpError(RuntimeError):
    """Raised when a request fails permanently."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class RateLimit:
    """A request budget.

    Attributes:
        per_second: Maximum sustained requests per second.
        per_hour: Maximum requests in any rolling hour.
    """

    per_second: float
    per_hour: int


class RateLimiter:
    """Blocking dual-window rate limiter.

    Enforces a minimum interval between requests *and* a rolling hourly cap. Thread-safe,
    because the API serves predictions concurrently while a background ingestion runs.
    """

    def __init__(self, limit: RateLimit) -> None:
        self._limit = limit
        self._min_interval = 1.0 / limit.per_second if limit.per_second > 0 else 0.0
        self._last_request: float = 0.0
        self._hour_window: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until another request is permitted by both windows."""
        while True:
            with self._lock:
                now = time.monotonic()
                wait = self._compute_wait(now)
                if wait <= 0:
                    self._last_request = now
                    self._hour_window.append(now)
                    return
            # Sleep outside the lock so concurrent callers are not serialised on it.
            time.sleep(wait)

    def _compute_wait(self, now: float) -> float:
        """Seconds to wait before the next request may be issued. Assumes the lock is held."""
        while self._hour_window and now - self._hour_window[0] >= _SECONDS_PER_HOUR:
            self._hour_window.popleft()

        wait = 0.0
        if self._min_interval > 0:
            wait = max(wait, self._last_request + self._min_interval - now)
        if len(self._hour_window) >= self._limit.per_hour:
            wait = max(wait, self._hour_window[0] + _SECONDS_PER_HOUR - now)
        return wait


class ResponseCache:
    """Content-addressed JSON response cache on the local filesystem.

    Entries marked immutable never expire: results of a completed season do not change, so
    a full backfill only ever pays for each request once.
    """

    def __init__(self, directory: Path, *, ttl_hours: float = 24.0, enabled: bool = True) -> None:
        self._directory = directory
        self._ttl_seconds = ttl_hours * _SECONDS_PER_HOUR
        self._enabled = enabled
        if enabled:
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key_for(url: str, params: dict[str, Any] | None) -> str:
        """Build a stable cache key for a request.

        Args:
            url: Fully qualified request URL.
            params: Query parameters, if any.

        Returns:
            A hex digest identifying the request.
        """
        canonical = url
        if params:
            canonical += "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        # Shard by prefix so directories stay small during a full backfill.
        return self._directory / key[:2] / f"{key}.json"

    def get(self, key: str, *, immutable: bool = False) -> Any | None:
        """Return a cached payload, or ``None`` on a miss or expiry.

        Args:
            key: Cache key from :meth:`key_for`.
            immutable: Treat the entry as never expiring.

        Returns:
            The cached payload, or ``None``.
        """
        if not self._enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None

        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A truncated cache file must not break ingestion; treat it as a miss.
            logger.debug("Discarding unreadable cache entry", extra={"key": key})
            return None

        if not immutable and (time.time() - entry.get("fetched_at", 0.0)) > self._ttl_seconds:
            return None
        return entry.get("payload")

    def set(self, key: str, payload: Any) -> None:
        """Store a payload.

        Args:
            key: Cache key from :meth:`key_for`.
            payload: JSON-serialisable response body.
        """
        if not self._enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write via a temporary file so a crash cannot leave a half-written entry.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"fetched_at": time.time(), "payload": payload}), encoding="utf-8"
        )
        tmp.replace(path)


class JsonHttpClient:
    """HTTP client returning parsed JSON, with rate limiting, caching and retries."""

    def __init__(
        self,
        base_url: str,
        *,
        rate_limit: RateLimit,
        cache: ResponseCache | None = None,
        timeout: float = 30.0,
        max_retries: int = 5,
        backoff_factor: float = 2.0,
        user_agent: str = "f1-champion-predictor/0.1",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._limiter = RateLimiter(rate_limit)
        self._cache = cache
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            follow_redirects=True,
        )
        self._requests_made = 0
        self._cache_hits = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._client.close()

    @property
    def stats(self) -> dict[str, int]:
        """Counters for requests actually issued and cache hits served."""
        return {"requests": self._requests_made, "cache_hits": self._cache_hits}

    def get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        immutable: bool = False,
    ) -> Any:
        """Fetch a URL and return the decoded JSON body.

        Args:
            path: Path appended to the base URL, or an absolute URL.
            params: Query parameters.
            immutable: Whether the response can be cached forever (true for completed
                seasons, whose results never change).

        Returns:
            The decoded JSON payload.

        Raises:
            HttpError: If the request fails after exhausting the retry budget.
        """
        url = path if path.startswith("http") else f"{self._base_url}/{path.lstrip('/')}"

        cache_key = ResponseCache.key_for(url, params)
        if self._cache is not None:
            cached = self._cache.get(cache_key, immutable=immutable)
            if cached is not None:
                self._cache_hits += 1
                return cached

        payload = self._request_with_retries(url, params)

        if self._cache is not None:
            self._cache.set(cache_key, payload)
        return payload

    def _request_with_retries(self, url: str, params: dict[str, Any] | None) -> Any:
        last_error: str = "no attempt was made"
        status: int | None = None

        for attempt in range(self._max_retries + 1):
            self._limiter.acquire()
            try:
                response = self._client.get(url, params=params)
                self._requests_made += 1
            except httpx.HTTPError as exc:
                last_error = f"transport error: {exc}"
                status = None
            else:
                if response.status_code < 400:
                    return response.json()

                status = response.status_code
                last_error = f"HTTP {status}"
                if status not in _RETRYABLE_STATUS:
                    raise HttpError(
                        f"Request to {url} failed with {last_error}", status_code=status
                    )
                self._sleep_before_retry(attempt, response)
                continue

            if attempt < self._max_retries:
                time.sleep(self._backoff_factor**attempt)

        msg = f"Request to {url} failed after {self._max_retries + 1} attempts ({last_error})"
        raise HttpError(msg, status_code=status)

    def _sleep_before_retry(self, attempt: int, response: httpx.Response) -> None:
        """Wait before retrying, preferring the server's ``Retry-After`` hint."""
        delay = self._backoff_factor**attempt
        if retry_after := response.headers.get("Retry-After"):
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                # Retry-After may be an HTTP date; the exponential fallback is fine.
                pass
        logger.warning(
            "Retrying request",
            extra={
                "status": response.status_code,
                "attempt": attempt + 1,
                "sleep_seconds": round(delay, 2),
            },
        )
        time.sleep(delay)
