"""
Tests for the owner lock and SSRF guard.

These cover the two ways a Slack bot with shell access gets taken over: an
untrusted user asking it to run code, and an untrusted *document* pointing it
at the cloud metadata service. Both defaults are fail-closed, which is the
property worth pinning — a safe default that depends on someone remembering
to set an environment variable is not a safe default.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import governor
import memory
import tools


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "memory.db")
    monkeypatch.setattr(tools, "_OWNER_WARNED", False)
    yield


# ── Owner lock ──

def test_unset_owner_fails_closed(monkeypatch):
    """
    Previously this failed OPEN: with no OWNER_SLACK_ID, every owner-only tool
    ran for any Slack user who could reach the bot — shell included.
    """
    monkeypatch.delenv("OWNER_SLACK_ID", raising=False)
    assert tools._is_owner("U_STRANGER") is False
    assert tools._is_owner("") is False


def test_owner_matches_only_the_configured_id(monkeypatch):
    monkeypatch.setenv("OWNER_SLACK_ID", "U_OWNER")
    assert tools._is_owner("U_OWNER") is True
    assert tools._is_owner("U_STRANGER") is False
    assert tools._is_owner("") is False


def test_owner_id_is_whitespace_tolerant(monkeypatch):
    monkeypatch.setenv("OWNER_SLACK_ID", "  U_OWNER  ")
    assert tools._is_owner("U_OWNER") is True


@pytest.mark.parametrize("tool_name", ["run_shell", "run_python"])
def test_code_execution_is_refused_for_non_owners(tool_name, monkeypatch):
    """The whole point: a stranger must not get RCE on the host."""
    monkeypatch.setenv("OWNER_SLACK_ID", "U_OWNER")
    result = tools.run_tool(tool_name, {
        "command" if tool_name == "run_shell" else "code": "echo pwned",
        "_requesting_user_id": "U_STRANGER",
    })
    assert result.startswith("❌ Not authorized")


def test_code_execution_is_refused_when_no_owner_is_configured(monkeypatch):
    monkeypatch.delenv("OWNER_SLACK_ID", raising=False)
    result = tools.run_tool("run_python", {"code": "print(1)", "_requesting_user_id": "U_ANYONE"})
    assert result.startswith("❌ Not authorized")


def test_the_owner_can_still_use_their_own_tools(monkeypatch):
    monkeypatch.setenv("OWNER_SLACK_ID", "U_OWNER")
    result = tools.run_tool("run_python", {"code": "print(6*7)", "_requesting_user_id": "U_OWNER"})
    assert "42" in result


def test_read_only_tools_are_open_to_everyone(monkeypatch):
    """Fail-closed applies to privilege, not to the whole bot."""
    monkeypatch.setenv("OWNER_SLACK_ID", "U_OWNER")
    for name in ("list_files", "list_tasks"):
        assert not tools.run_tool(name, {}).startswith("❌ Not authorized")


def test_every_owner_only_tool_actually_exists():
    assert set(tools.OWNER_ONLY_TOOLS) <= set(tools.TOOLS)


def test_autonomy_tools_are_owner_gated():
    """Committing shared workers and a metered paid route is the owner's call."""
    for name in ("start_background_run", "schedule_task", "cancel_schedule"):
        assert name in tools.OWNER_ONLY_TOOLS, name


def test_run_python_output_is_redacted(monkeypatch):
    """Model-written code can print a token as easily as run_shell can."""
    monkeypatch.setenv("OWNER_SLACK_ID", "U_OWNER")
    result = tools.run_tool("run_python", {
        "code": "print('ghp_' + 'a' * 36)",
        "_requesting_user_id": "U_OWNER",
    })
    assert "ghp_aaaa" not in result


@pytest.mark.parametrize("secret", [
    "ghp_" + "a" * 36,
    "github_pat_" + "B" * 30,
    "xoxb-1234567890-abcdefghij",
    "sk-" + "c" * 32,
    "AIza" + "d" * 35,
    "mg_" + "e" * 24,
])
def test_bare_credentials_are_redacted_without_a_keyword(secret):
    """
    `echo $GITHUB_TOKEN` prints a token with no keyword beside it. The
    keyword-based patterns miss that entirely, which is the most likely way a
    real credential from this box reaches a Slack channel.
    """
    assert secret not in tools._redact_shell_output(f"here it is: {secret}")


def test_redaction_leaves_ordinary_output_alone():
    output = "3 files changed, 42 insertions(+)\nAll tests passed."
    assert tools._redact_shell_output(output) == output


# ── SSRF ──

@pytest.mark.parametrize("host", [
    "localhost",
    "127.0.0.1",
    "169.254.169.254",           # AWS/GCP instance metadata
    "metadata.google.internal",
    "10.0.0.1",
    "192.168.1.1",
    "172.16.0.1",
    "0.0.0.0",
    "something.internal",
    "printer.local",
    "",
])
def test_internal_hosts_are_blocked(host):
    assert tools._url_host_is_safe(host) is False


def test_public_hosts_are_allowed():
    assert tools._url_host_is_safe("example.com") is True


def test_unresolvable_hosts_are_blocked():
    assert tools._url_host_is_safe("no-such-host.invalid") is False


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
    "http://metadata.google.internal/computeMetadata/v1/",
    "http://localhost:8080/admin",
    "http://127.0.0.1/",
])
def test_fetch_url_refuses_internal_targets(url):
    result = tools.fetch_url(url)
    assert result.startswith("❌ Blocked")
    assert "private, loopback" in result


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "gopher://evil/",
    "ftp://internal/",
])
def test_fetch_url_refuses_non_http_schemes(url):
    result = tools.fetch_url(url)
    assert result.startswith("❌ Blocked")
    assert "scheme" in result


def test_redirects_are_revalidated(monkeypatch):
    """
    The subtle hole: a public URL that 302s to the metadata service. Checking
    only the first URL and letting requests follow redirects would sail
    straight through it.
    """
    hops = []

    class FakeResponse:
        def __init__(self, status, location=None):
            self.status_code = status
            self.headers = {"Location": location} if location else {}
            self.text = "ok"

    def fake_get(url, **kwargs):
        hops.append(url)
        assert kwargs.get("allow_redirects") is False, "redirects must not be auto-followed"
        if "/redirect" in url:
            return FakeResponse(302, "http://169.254.169.254/latest/meta-data/")
        return FakeResponse(200)

    monkeypatch.setattr(tools.http_requests, "get", fake_get)
    # The first hop is a genuinely public host, so it passes the initial
    # check — the danger is entirely in where it sends us next.
    response, error = tools._safe_http_get("http://example.com/redirect", {})

    assert response is None
    assert "private, loopback" in error
    assert hops == ["http://example.com/redirect"]   # never fetched the metadata URL


def test_redirect_chains_terminate(monkeypatch):
    class FakeResponse:
        def __init__(self):
            self.status_code = 302
            self.headers = {"Location": "https://example.com/next"}

    monkeypatch.setattr(tools.http_requests, "get", lambda url, **kw: FakeResponse())
    response, error = tools._safe_http_get("https://example.com/", {})
    assert response is None
    assert "too many redirects" in error


# ── The two axes stay separate ──

def test_external_tools_are_all_owner_only():
    for name in governor.external_tools():
        assert name in tools.OWNER_ONLY_TOOLS, f"{name} reaches outside but is not owner-gated"


def test_shell_is_owner_only_but_not_external():
    """
    Deliberate: gating `ls` behind a Slack approval in every scheduled task is
    how an approval queue turns into a button people press without reading.
    """
    assert "run_shell" in tools.OWNER_ONLY_TOOLS
    assert "run_shell" not in governor.external_tools()
