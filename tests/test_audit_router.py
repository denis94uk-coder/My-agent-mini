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
