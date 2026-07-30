"""Tests for the rate-limited, cached HTTP client."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from f1cp.data.http import HttpError, JsonHttpClient, RateLimit, RateLimiter, ResponseCache

pytestmark = pytest.mark.unit

BASE_URL = "https://api.example.test/f1"


class TestRateLimiter:
    def test_spaces_requests_by_the_per_second_budget(self) -> None:
        limiter = RateLimiter(RateLimit(per_second=100.0, per_hour=1_000_000))
        clock = 0.0
        limiter._last_request = clock
        limiter._hour_window.append(clock)

        # 100 req/s means a 10 ms floor between consecutive requests.
        assert limiter._compute_wait(clock) == pytest.approx(0.01, abs=1e-6)
        assert limiter._compute_wait(clock + 0.01) <= 0

    def test_blocks_once_the_hourly_budget_is_exhausted(self) -> None:
        limiter = RateLimiter(RateLimit(per_second=1000.0, per_hour=3))
        for offset in (0.0, 0.1, 0.2):
            limiter._hour_window.append(offset)
        limiter._last_request = 0.2

        # The fourth request must wait until the oldest falls out of the hour window.
        wait = limiter._compute_wait(1.0)
        assert wait == pytest.approx(3599.0, abs=1e-3)

    def test_expired_entries_leave_the_window(self) -> None:
        limiter = RateLimiter(RateLimit(per_second=1000.0, per_hour=1))
        limiter._hour_window.append(0.0)
        assert limiter._compute_wait(3601.0) <= 0
        assert not limiter._hour_window


class TestResponseCache:
    def test_round_trips_a_payload(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path, ttl_hours=1.0)
        key = ResponseCache.key_for("https://x.test/a", {"limit": 100})
        cache.set(key, {"value": 1})
        assert cache.get(key) == {"value": 1}

    def test_key_is_order_independent(self) -> None:
        first = ResponseCache.key_for("https://x.test", {"a": 1, "b": 2})
        second = ResponseCache.key_for("https://x.test", {"b": 2, "a": 1})
        assert first == second

    def test_entries_expire(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path, ttl_hours=0.0)
        key = "k"
        cache.set(key, {"value": 1})
        assert cache.get(key) is None

    def test_immutable_entries_never_expire(self, tmp_path: Path) -> None:
        """Results of a completed season do not change, so they are cached forever."""
        cache = ResponseCache(tmp_path, ttl_hours=0.0)
        key = "k"
        cache.set(key, {"value": 1})
        assert cache.get(key, immutable=True) == {"value": 1}

    def test_corrupt_entry_is_treated_as_a_miss(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path, ttl_hours=1.0)
        key = ResponseCache.key_for("https://x.test/a", None)
        cache.set(key, {"value": 1})
        cache._path(key).write_text("{ truncated", encoding="utf-8")
        assert cache.get(key) is None

    def test_disabled_cache_stores_nothing(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path, enabled=False)
        cache.set("k", {"value": 1})
        assert cache.get("k") is None


class TestJsonHttpClient:
    @respx.mock
    def test_returns_decoded_json(self) -> None:
        respx.get(f"{BASE_URL}/races").mock(return_value=httpx.Response(200, json={"ok": True}))
        with JsonHttpClient(BASE_URL, rate_limit=RateLimit(1000.0, 100_000)) as client:
            assert client.get_json("races") == {"ok": True}

    @respx.mock
    def test_serves_repeat_requests_from_cache(self, tmp_path: Path) -> None:
        route = respx.get(f"{BASE_URL}/races").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        cache = ResponseCache(tmp_path, ttl_hours=1.0)
        with JsonHttpClient(BASE_URL, rate_limit=RateLimit(1000.0, 100_000), cache=cache) as client:
            client.get_json("races")
            client.get_json("races")

        assert route.call_count == 1
        assert client.stats == {"requests": 1, "cache_hits": 1}

    @respx.mock
    def test_retries_transient_failures_then_succeeds(self) -> None:
        respx.get(f"{BASE_URL}/races").mock(
            side_effect=[
                httpx.Response(503),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        with JsonHttpClient(
            BASE_URL, rate_limit=RateLimit(1000.0, 100_000), backoff_factor=0.0
        ) as client:
            assert client.get_json("races") == {"ok": True}

    @respx.mock
    def test_does_not_retry_client_errors(self) -> None:
        route = respx.get(f"{BASE_URL}/nope").mock(return_value=httpx.Response(404))
        with (
            JsonHttpClient(BASE_URL, rate_limit=RateLimit(1000.0, 100_000)) as client,
            pytest.raises(HttpError) as excinfo,
        ):
            client.get_json("nope")

        assert excinfo.value.status_code == 404
        assert route.call_count == 1

    @respx.mock
    def test_gives_up_after_the_retry_budget(self) -> None:
        respx.get(f"{BASE_URL}/races").mock(return_value=httpx.Response(500))
        with (
            JsonHttpClient(
                BASE_URL, rate_limit=RateLimit(1000.0, 100_000), max_retries=2, backoff_factor=0.0
            ) as client,
            pytest.raises(HttpError, match="after 3 attempts"),
        ):
            client.get_json("races")
