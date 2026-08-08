"""
Phase 3.10 (partial) — provider router accounting.

Carried forward from the Phase 2 working set: the parts of the router that can
be tested without importing bot.py. The rest of area 10 (cooldown application
and recovery, all-routes-down, expired key vs keyless) lands with Phase 3.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_support import audit_env  # noqa: F401

import governor


def test_paid_routes_sort_last_whatever_their_registration_order(audit_env):
    """Registration order is just block order in build_providers; the sort is
    what stops a free route being skipped for a paid one."""
    names = ["Merge Gateway", "Pollinations (keyless)", "NVIDIA", "Groq"]
    ordered = sorted(names, key=lambda n: (1 if governor.is_paid(n) else 0,
                                           governor.route_order_rank(n)))
    assert ordered[-1] == "Merge Gateway"
    assert governor.is_paid("Pollinations (keyless)") is False


def test_paid_budget_exhaustion_leaves_free_routes_serving(audit_env, monkeypatch):
    monkeypatch.setenv("PAID_DAILY_LIMIT", "1")
    governor.record_ai_call("Merge Gateway", input_chars=1, output_chars=1)
    assert governor.paid_budget_exhausted() is True
    assert governor.is_paid("Groq") is False


@pytest.mark.parametrize("headers,expected", [
    ({"Retry-After": "12"}, 13),
    ({"x-ratelimit-reset-tokens": "8s"}, 9),
    ({"x-ratelimit-reset-requests": "2m59.56s"}, 180.56),
    ({}, None),
])
def test_429_reset_window_is_read_not_guessed(audit_env, headers, expected):
    """A 429 states its own window. The HTTP-date form must not have its digits
    read as a duration."""
    from requests.structures import CaseInsensitiveDict

    class Resp:
        status_code = 429

        def __init__(self, h):
            self.headers = CaseInsensitiveDict(h)

    got = governor.retry_after_seconds(Resp(headers), 90)
    if expected is None:
        assert got == 90, "a missing header must fall back to the default"
    else:
        assert abs(got - expected) < 1.0, (headers, got)


def test_an_http_date_retry_after_is_not_parsed_as_a_duration(audit_env):
    from requests.structures import CaseInsensitiveDict

    class Resp:
        status_code = 429
        headers = CaseInsensitiveDict({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})

    got = governor.retry_after_seconds(Resp(), 90)
    assert got != 21, "the date's digits were read as seconds"
    assert 0 <= got <= 300


# ── cooldown, exhaustion and degradation (needs bot.py's router) ──

from audit_support import slack_bot  # noqa: E402,F401


class _Resp:
    def __init__(self, status, headers=None):
        from requests.structures import CaseInsensitiveDict
        self.status_code = status
        self.headers = CaseInsensitiveDict(headers or {})


def _err(status, headers=None):
    exc = Exception(f"HTTP {status}")
    exc.response = _Resp(status, headers)
    return exc


@pytest.mark.parametrize("status,headers", [
    (429, {"Retry-After": "30"}),
    (429, {}),
    (500, {}),
    (503, {}),
    (None, {}),          # timeout / connection error: no response attached
])
def test_a_failure_puts_the_route_on_cooldown(audit_env, slack_bot, status, headers):
    bot, _ = slack_bot
    provider = {"name": "Groq", "type": "openai_compat", "model": "m", "url": "u"}
    bot.PROVIDER_HEALTH.clear()
    error = _err(status, headers) if status else TimeoutError("read timed out")
    bot._record_provider_failure(provider, error)
    assert bot._provider_is_available(provider) is False
    assert "cooldown" in bot._provider_status(provider)


def test_a_route_recovers_when_its_cooldown_expires(audit_env, slack_bot):
    import time
    bot, _ = slack_bot
    provider = {"name": "Groq", "type": "openai_compat", "model": "m", "url": "u"}
    bot.PROVIDER_HEALTH.clear()
    bot._record_provider_failure(provider, _err(429, {"Retry-After": "30"}))
    assert bot._provider_is_available(provider) is False
    bot.PROVIDER_HEALTH["Groq"]["cooldown_until"] = time.time() - 1
    assert bot._provider_is_available(provider) is True
    bot._record_provider_success(provider)
    assert bot._provider_status(provider).startswith("healthy")


def test_a_429_that_names_its_window_is_honoured_over_the_default(audit_env, slack_bot):
    import time
    bot, _ = slack_bot
    provider = {"name": "Groq", "type": "openai_compat", "model": "m", "url": "u"}
    bot.PROVIDER_HEALTH.clear()
    bot._record_provider_failure(provider, _err(429, {"Retry-After": "5"}))
    remaining = bot.PROVIDER_HEALTH["Groq"]["cooldown_until"] - time.time()
    assert 4 <= remaining <= 8, f"ignored the stated window: {remaining}s"


def test_all_routes_down_degrades_cleanly_without_a_retry_storm(audit_env, slack_bot, monkeypatch):
    bot, _ = slack_bot
    attempts = []

    def always_fails(url, **kw):
        attempts.append(url)
        raise ConnectionError("network unreachable")

    bot.PROVIDERS = [
        {"name": "Pollinations (keyless)", "type": "openai_compat", "api_key": "",
         "model": "openai-fast", "url": "https://text.pollinations.ai/{model}", "keyless": True},
        {"name": "Groq", "type": "openai_compat", "api_key": "k", "model": "m",
         "url": "https://api.groq.com/openai/v1/chat/completions"},
    ]
    bot.PROVIDER_HEALTH.clear()
    monkeypatch.setattr(bot.http_requests, "post", always_fails)
    monkeypatch.setattr(bot.http_requests, "get", always_fails)

    out = bot._real_call_ai([{"role": "user", "content": "hi"}], "sys")
    assert out.startswith("❌"), out[:120]
    assert "All AI providers failed" in out
    assert len(attempts) == 2, f"each route must be tried once, not retried: {attempts}"

    # Second call while everything is cooling down: no further network attempts.
    attempts.clear()
    again = bot._real_call_ai([{"role": "user", "content": "hi"}], "sys")
    assert attempts == [], "hammered routes that were already cooling down"
    assert "cooling down" in again


def test_an_unusable_key_route_does_not_block_the_keyless_route(audit_env, slack_bot, monkeypatch):
    """A stale key left in .env must never delay or block the free route."""
    bot, _ = slack_bot
    served = []

    def post(url, **kw):
        if "groq" in url:
            raise _err(401)
        served.append(url)

        class OK:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "free answer"}}]}
            text = "free answer"
        return OK()

    bot.PROVIDERS = [
        {"name": "Pollinations (keyless)", "type": "openai_compat", "api_key": "",
         "model": "openai-fast", "url": "https://text.pollinations.ai/chat", "keyless": True},
        {"name": "Groq", "type": "openai_compat", "api_key": "expired", "model": "m",
         "url": "https://api.groq.com/openai/v1/chat/completions"},
    ]
    bot.PROVIDER_HEALTH.clear()
    monkeypatch.setattr(bot.http_requests, "post", post)
    out = bot._real_call_ai([{"role": "user", "content": "hi"}], "sys")
    assert out == "free answer"
    assert served and "pollinations" in served[0]


def test_keyless_is_registered_first_by_default(audit_env, slack_bot, monkeypatch):
    bot, _ = slack_bot
    monkeypatch.setenv("POLLINATIONS_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaFAKE")
    monkeypatch.setenv("MERGE_GATEWAY_API_KEY", "mg_fake")
    bot.build_providers()
    names = [p["name"] for p in bot.PROVIDERS]
    assert names[0].startswith("Pollinations"), names
    assert names[-1] == "Merge Gateway", names
