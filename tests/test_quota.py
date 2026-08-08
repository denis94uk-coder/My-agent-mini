"""
Rate-limit awareness.

The bot's own logs drove these: a route with a 12,000 token/minute ceiling
served one call of a 10-iteration agent loop and rejected the next, because
each call re-sends the whole system prompt. The daily budget was ~95% unspent
at the time. Two defects met there — nothing tracked the minute window, and a
429 parked the route for a flat 90 seconds when the window clears in single
digits.
"""

import time

import pytest

import governor


@pytest.fixture(autouse=True)
def _clean_window(monkeypatch):
    governor.reset_rate_limit_state()
    monkeypatch.delenv("ROUTER_TPM_LIMITS", raising=False)
    yield
    governor.reset_rate_limit_state()


# ── Limits ──

def test_known_free_tiers_have_a_default_limit():
    assert governor.tpm_limit("Groq") == 12_000


def test_unknown_routes_are_unmetered():
    """A limit nobody configured must not throttle traffic on a guess."""
    assert governor.tpm_limit("Pollinations (keyless)") == 0
    assert governor.tpm_limit("") == 0


def test_env_overrides_and_adds_routes(monkeypatch):
    monkeypatch.setenv("ROUTER_TPM_LIMITS", "groq:99,gemini:250000")
    assert governor.tpm_limit("Groq") == 99
    assert governor.tpm_limit("Gemini") == 250_000


def test_limits_match_by_substring_so_renames_survive(monkeypatch):
    """build_providers names routes for humans; matching must not be brittle."""
    monkeypatch.setenv("ROUTER_TPM_LIMITS", "groq:5000")
    assert governor.tpm_limit("Groq (llama-3.3-70b)") == 5000


def test_garbage_in_the_override_is_ignored(monkeypatch):
    monkeypatch.setenv("ROUTER_TPM_LIMITS", "groq:notanumber,,:,broken")
    assert governor.tpm_limit("Groq") == 12_000


# ── The rolling window ──

def test_tokens_accumulate_within_the_minute():
    governor.record_tokens("Groq", 1000)
    governor.record_tokens("Groq", 500)
    assert governor.tokens_last_minute("Groq") == 1500


def test_spend_older_than_a_minute_rolls_off(monkeypatch):
    governor.record_tokens("Groq", 9000)
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now + 61)
    assert governor.tokens_last_minute("Groq") == 0


def test_routes_are_tracked_separately():
    governor.record_tokens("Groq", 9000)
    assert governor.tokens_last_minute("Gemini") == 0


# ── The wait calculation ──

def test_no_wait_while_the_window_has_headroom():
    governor.record_tokens("Groq", 1000)
    assert governor.tpm_wait_seconds("Groq", 4400) == 0


def test_unmetered_routes_never_wait():
    governor.record_tokens("Pollinations (keyless)", 10**9)
    assert governor.tpm_wait_seconds("Pollinations (keyless)", 50_000) == 0


def test_wait_is_returned_once_the_window_is_full():
    """The exact case from the logs: two calls fit, the third must hold."""
    governor.record_tokens("Groq", 4400)
    governor.record_tokens("Groq", 4400)
    wait = governor.tpm_wait_seconds("Groq", 4400)
    assert 0 < wait <= 61


def test_wait_expires_with_the_oldest_entry(monkeypatch):
    governor.record_tokens("Groq", 11_000)
    now = time.time()
    # 50s in, the entry has 10s left on its minute.
    monkeypatch.setattr(time, "time", lambda: now + 50)
    wait = governor.tpm_wait_seconds("Groq", 4400)
    assert 9 <= wait <= 12


def test_a_call_larger_than_the_whole_limit_does_not_wait():
    """
    Waiting cannot make room for it, so the honest 429 is more useful than a
    pause that changes nothing — and this must not become an infinite hold.
    """
    assert governor.tpm_wait_seconds("Groq", 20_000) == 0


# ── Backing off for exactly as long as the provider asked ──

class _Response:
    def __init__(self, headers):
        self.headers = headers
        self.status_code = 429


def test_plain_retry_after_seconds_is_used():
    assert governor.retry_after_seconds(_Response({"retry-after": "8"}), 90) == 9


def test_groq_style_duration_is_parsed():
    """Groq sends x-ratelimit-reset-tokens as "7.66s", not a bare number."""
    got = governor.retry_after_seconds(_Response({"x-ratelimit-reset-tokens": "7.66s"}), 90)
    assert 8 <= got <= 9


def test_compound_duration_is_parsed():
    got = governor.retry_after_seconds(_Response({"retry-after": "2m59.56s"}), 90)
    assert 180 <= got <= 181


def test_a_missing_header_keeps_the_configured_default():
    """Absent guidance must not be read as "retry immediately"."""
    assert governor.retry_after_seconds(_Response({}), 90) == 90
    assert governor.retry_after_seconds(None, 90) == 90
    assert governor.retry_after_seconds(_Response({"retry-after": "  "}), 90) == 90


def test_a_short_window_beats_the_flat_cooldown():
    """
    The defect this closes: a token/minute window clearing in 8s parked the
    route for 90s, with the whole daily budget still unspent.
    """
    assert governor.retry_after_seconds(_Response({"retry-after": "8"}), 90) < 90


def test_a_long_window_is_honoured_past_the_default():
    """A daily exhaustion needs longer than 90s, so the header wins upward."""
    assert governor.retry_after_seconds(_Response({"retry-after": "240"}), 90) > 90


def test_the_wait_is_capped():
    """One header must not park a route for an hour."""
    assert governor.retry_after_seconds(_Response({"retry-after": "86400"}), 90) <= 300


def test_unparseable_headers_fall_back_rather_than_raise():
    for value in ("soon", "-", "???"):
        assert governor.retry_after_seconds(_Response({"retry-after": value}), 90) == 90


def test_http_date_form_is_not_read_as_a_duration():
    """
    Retry-After also permits an HTTP-date (RFC 9110). Summing its digits as a
    duration read a 2015 timestamp as ~2,071 seconds and parked the route for
    the full cap.
    """
    past = _Response({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"})
    assert governor.retry_after_seconds(past, 90) == 0.0


def test_http_date_in_the_future_waits_until_then():
    from email.utils import formatdate
    soon = _Response({"retry-after": formatdate(time.time() + 30, usegmt=True)})
    assert 25 <= governor.retry_after_seconds(soon, 90) <= 31


# ── Estimation ──

def test_estimate_includes_an_output_allowance():
    """Charging input only undercounts, and undercounting spends a 429."""
    assert governor.estimate_tokens(17_600) > 17_600 // 4


def test_estimate_handles_empty_and_negative_input():
    assert governor.estimate_tokens(0) > 0
    assert governor.estimate_tokens(-5) > 0
