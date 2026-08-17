"""
The health endpoint.

`server_health` is a tool, which means the only way to ask whether the bot is
alive is to ask the bot — useless in the one case that matters. The
`ops-watch` workflow shares the blind spot: it runs inside the process it is
watching. This endpoint answers from outside the agent loop.

The properties that make it worth having, rather than a second thing that can
lie: a broken subsystem must not hide the healthy ones, "cannot do its job"
must be distinguishable from "not running", and it must never be able to take
down the process it reports on.
"""

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import health
import memory
import runner


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "memory.db")
    monkeypatch.setenv("HEALTH_PORT", "0")  # let the OS pick a free port
    yield
    health.stop()


@pytest.fixture
def endpoint(monkeypatch):
    """A live server; returns a (path) -> (status, body) caller."""
    assert health.start(), "health server did not bind"
    port = health._SERVER.server_port

    def call(path="/healthz"):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{path}", timeout=5
            ) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    return call


# ── The report ──

def test_the_snapshot_covers_every_subsystem():
    checks = health.snapshot()["checks"]

    assert set(checks) >= {"database", "workers", "runs", "routes", "sandbox"}


def test_the_database_is_actually_queried():
    assert health.snapshot()["checks"]["database"] == "ok"


def test_a_broken_probe_does_not_hide_the_healthy_ones(monkeypatch):
    """INVARIANT: one raising probe must not take out the whole report — that
    is the opposite of what a health check is for."""
    def explode():
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(memory, "get_db", explode)

    report = health.snapshot()

    assert "unknown" in report["checks"]["database"]
    assert report["checks"]["sandbox"]  # still reported
    assert report["status"] == health.STATUS_DEGRADED


def test_snapshot_never_raises(monkeypatch):
    """It runs inside the process it reports on. Raising here would let the
    health check kill the thing it is monitoring."""
    monkeypatch.setattr(
        runner, "worker_count", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    assert health.snapshot()["status"] in (health.STATUS_OK, health.STATUS_DEGRADED)


# ── Readiness vs liveness ──

def test_a_database_failure_is_degraded(monkeypatch):
    monkeypatch.setattr(
        memory, "get_db", lambda: (_ for _ in ()).throw(RuntimeError("locked"))
    )

    report = health.snapshot()

    assert report["status"] == health.STATUS_DEGRADED
    assert any("database" in r for r in report["reasons"])


def test_dead_workers_are_degraded_once_they_were_started(monkeypatch):
    """A worker thread that died takes its share of the queue with it and
    nothing else notices — runs simply stop draining."""
    monkeypatch.setattr(runner, "workers_started", lambda: True)
    monkeypatch.setattr(runner, "worker_count", lambda: 2)

    report = health.snapshot()

    assert report["status"] == health.STATUS_DEGRADED
    assert any("workers" in r for r in report["reasons"])


def test_workers_that_were_never_started_are_not_a_fault(monkeypatch):
    """INVARIANT: 'never started' is normal for a process running without the
    run engine. Reporting it as degraded trains the reader to ignore this."""
    monkeypatch.setattr(runner, "workers_started", lambda: False)

    report = health.snapshot()

    assert not any("workers" in r for r in report.get("reasons", []))


def test_live_workers_are_counted(monkeypatch):
    stop = threading.Event()
    threads = [
        threading.Thread(
            target=stop.wait, name=f"{runner.WORKER_THREAD_PREFIX}{i}", daemon=True
        )
        for i in (1, 2)
    ]
    [t.start() for t in threads]
    monkeypatch.setattr(runner, "worker_count", lambda: 2)
    monkeypatch.setattr(runner, "workers_started", lambda: True)

    try:
        report = health.snapshot()
        assert report["checks"]["workers"]["alive"] == 2
        assert not any("workers" in r for r in report.get("reasons", []))
    finally:
        stop.set()
        [t.join(timeout=5) for t in threads]


def test_the_worker_thread_name_matches_what_runner_uses():
    """The probe counts threads by name prefix. If runner renames them the
    endpoint silently reports zero live workers and pages forever."""
    source = Path(runner.__file__).read_text()

    assert "name=f\"{WORKER_THREAD_PREFIX}" in source


def test_no_routes_is_degraded(monkeypatch):
    """The process is up and cannot serve anybody — exactly the state that
    should page, and exactly the one a naive 'is the process running' check
    reports as fine."""
    monkeypatch.setattr(health, "_ROUTE_SOURCE", lambda: [])

    report = health.snapshot()

    assert report["status"] == health.STATUS_DEGRADED
    assert any("routes" in r for r in report["reasons"])


def test_unreported_routes_are_not_the_same_as_no_routes():
    """INVARIANT: nobody having told us is not a fault. Treating it as one
    marks the process degraded for a reason that is not true."""
    assert health._ROUTE_SOURCE is None  # nothing registered in tests

    report = health.snapshot()

    assert not any("routes" in r for r in report.get("reasons", []))


def test_health_does_not_import_bot():
    """INVARIANT: bot.py owns the Slack client and exits at import when its
    tokens are absent. Every other module has to stay importable without
    them, which is what makes the whole test suite possible."""
    source = Path(health.__file__).read_text()

    assert "import bot" not in source


# ── Over HTTP ──

def test_healthz_answers(endpoint):
    status, body = endpoint("/healthz")

    assert status in (200, 503)
    assert "uptime_seconds" in body


def test_degraded_returns_503_so_curl_f_fails(endpoint, monkeypatch):
    """So a systemd or uptime check fails without parsing the body."""
    monkeypatch.setattr(
        memory, "get_db", lambda: (_ for _ in ()).throw(RuntimeError("down"))
    )

    status, body = endpoint("/healthz")

    assert status == 503
    assert body["status"] == health.STATUS_DEGRADED


def test_livez_never_fails_on_readiness(endpoint, monkeypatch):
    """INVARIANT: liveness and readiness are different questions. A supervisor
    restarting a bot that is merely out of AI routes would lose every queued
    run to no purpose."""
    monkeypatch.setattr(
        memory, "get_db", lambda: (_ for _ in ()).throw(RuntimeError("down"))
    )

    status, body = endpoint("/livez")

    assert status == 200
    assert body["status"] == health.STATUS_OK


def test_an_unknown_path_is_404(endpoint):
    status, _ = endpoint("/../../etc/passwd")

    assert status == 404


# ── Binding ──

def test_it_binds_loopback_by_default(monkeypatch):
    """The report names MCP servers, routes and run counts. There is no
    reverse proxy or auth in front of this box."""
    monkeypatch.delenv("HEALTH_HOST", raising=False)

    assert health._host() == "127.0.0.1"


def test_it_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("HEALTH_ENABLED", "0")

    assert health.start() is False


def test_a_taken_port_does_not_stop_the_bot(monkeypatch):
    """Failing to bind is not a reason to refuse to run the agent."""
    blocker = health.HTTPServer(("127.0.0.1", 0), health._Handler)
    monkeypatch.setenv("HEALTH_PORT", str(blocker.server_port))

    try:
        assert health.start() is False
    finally:
        blocker.server_close()


def test_starting_twice_is_a_no_op(endpoint):
    assert health.start() is False
