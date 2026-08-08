"""
A localhost health endpoint.

`server_health` already exists as a *tool*, which means the only way to ask
whether the bot is alive is to ask the bot — useless in exactly the case that
matters. If the Slack socket has dropped, a worker thread has died, or the
process is wedged, the tool cannot answer and the silence is indistinguishable
from nobody having asked. The `ops-watch` workflow has the same blind spot: it
runs inside the process it is meant to be watching.

This binds a tiny HTTP server so something outside the process — systemd, a
cron curl, an uptime check — can get an answer that does not depend on the
agent loop being healthy enough to produce one.

Deliberately small:

  • **Loopback by default.** The response names configured MCP servers,
    provider routes and run counts. That is reconnaissance for anyone who can
    reach it, and this box has no reverse proxy or auth layer in front of it.
    Binding elsewhere takes an explicit HEALTH_HOST.
  • **stdlib `http.server`, one thread.** A framework would be a dependency
    and a worker pool for an endpoint answering a request a minute.
  • **Never raises into the caller.** A failing health check must not be able
    to take down the process it is reporting on, so every probe is guarded and
    degrades to "unknown" rather than propagating.
"""

import os
import json
import time
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger("my-agent-mini")

STARTED_AT = time.time()

# Liveness and readiness are different questions and conflating them makes the
# endpoint useless for both. The process answering at all is liveness; whether
# it can actually do work is readiness, and a bot with no AI routes left is
# running perfectly while being unable to serve anybody.
STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"


def _host() -> str:
    return os.getenv("HEALTH_HOST", "127.0.0.1").strip() or "127.0.0.1"


def _port() -> int:
    try:
        return int(os.getenv("HEALTH_PORT", "8787"))
    except ValueError:
        return 8787


def enabled() -> bool:
    return os.getenv("HEALTH_ENABLED", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


# Reporting the AI routes needs bot.py's PROVIDERS, and bot.py owns the Slack
# client. Keeping the dependency pointing that way — bot tells health, health
# never imports bot — is what lets every other module stay importable in tests
# with no Slack tokens present.
_ROUTE_SOURCE = None


def set_route_source(fn) -> None:
    """Register a callable returning the current AI route names."""
    global _ROUTE_SOURCE
    _ROUTE_SOURCE = fn


def _route_names() -> list:
    if _ROUTE_SOURCE is None:
        # Not "no routes" — nobody told us. Reporting an empty list would mark
        # the process degraded for a reason that is not true.
        return ["(not reported)"]
    return _ROUTE_SOURCE() or []


def _probe(name: str, fn):
    """Run one probe, converting any failure into a reported value.

    A probe that raises would otherwise take out the whole report, which means
    one broken subsystem hides every healthy one — the opposite of what a
    health check is for.
    """
    try:
        return fn()
    except Exception as e:
        logger.debug(f"Health probe {name} failed: {e}")
        return f"unknown ({str(e)[:80]})"


def snapshot() -> dict:
    """Everything the endpoint reports. Safe to call at any time."""
    report = {
        "status": STATUS_OK,
        "uptime_seconds": int(time.time() - STARTED_AT),
        "checks": {},
    }
    checks = report["checks"]

    def check_database():
        import memory
        conn = memory.get_db()
        try:
            conn.execute("SELECT 1").fetchone()
            return "ok"
        finally:
            conn.close()

    def check_workers():
        import runner
        return {
            "configured": runner.worker_count(),
            "alive": sum(
                1 for t in threading.enumerate()
                if t.name.startswith(runner.WORKER_THREAD_PREFIX) and t.is_alive()
            ),
            "started": runner.workers_started(),
        }

    def check_runs():
        counts = {}
        conn = None
        try:
            import memory
            conn = memory.get_db()
            for status, count in conn.execute(
                "SELECT status, COUNT(*) FROM runs GROUP BY status"
            ).fetchall():
                counts[status] = count
        finally:
            if conn:
                conn.close()
        return counts

    def check_routes():
        # Pushed in by bot.py rather than imported from it. Importing bot here
        # would drag the Slack client into a module the tests need to import
        # without one — and bot.py exits at import when its tokens are absent,
        # which is a SystemExit, which `_probe` deliberately does not catch.
        return list(_route_names())

    def check_sandbox():
        import isolation
        return isolation.sandbox_status()

    checks["database"] = _probe("database", check_database)
    checks["workers"] = _probe("workers", check_workers)
    checks["runs"] = _probe("runs", check_runs)
    checks["routes"] = _probe("routes", check_routes)
    checks["sandbox"] = _probe("sandbox", check_sandbox)

    # Readiness. Each of these means the process is up but cannot do its job,
    # which is the state worth paging on — a wedged bot that still answers
    # "ok" is why this endpoint exists.
    reasons = []
    if checks["database"] != "ok":
        reasons.append("database unreachable")
    workers = checks["workers"]
    # Only a fault once the workers were actually started. "Never started" is
    # a normal state for a process running without the run engine, and
    # reporting it as degraded would train whoever reads this to ignore it.
    if (
        isinstance(workers, dict)
        and workers.get("started")
        and workers["alive"] < workers["configured"]
    ):
        reasons.append(
            f"{workers['configured'] - workers['alive']} of "
            f"{workers['configured']} run workers not alive"
        )
    if isinstance(checks["routes"], list) and not checks["routes"]:
        reasons.append("no AI routes configured")

    if reasons:
        report["status"] = STATUS_DEGRADED
        report["reasons"] = reasons
    return report


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        # The default handler writes every request to stderr, which for a
        # once-a-minute uptime check is pure log noise.
        pass

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"

        if path in ("/healthz", "/", "/health"):
            report = snapshot()
            # 503 when degraded so a plain `curl -f` or a systemd/uptime check
            # fails without having to parse the body.
            self._respond(200 if report["status"] == STATUS_OK else 503, report)
        elif path == "/livez":
            # Liveness only: the process is running and can answer. Never 503,
            # or a supervisor restarts a bot that is merely out of AI routes.
            self._respond(200, {"status": STATUS_OK,
                                "uptime_seconds": int(time.time() - STARTED_AT)})
        else:
            self._respond(404, {"error": "not found"})

    def _respond(self, status: int, payload: dict):
        body = json.dumps(payload, indent=2, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


_SERVER = None


def start() -> bool:
    """Start the health server in a daemon thread. Returns whether it bound.

    Failure to bind is logged and swallowed: the port being taken is not a
    reason to refuse to run the bot.
    """
    global _SERVER
    if not enabled() or _SERVER is not None:
        return False

    try:
        _SERVER = HTTPServer((_host(), _port()), _Handler)
    except OSError as e:
        logger.warning(f"Health endpoint could not bind {_host()}:{_port()}: {e}")
        return False

    threading.Thread(
        target=_SERVER.serve_forever, name="health-server", daemon=True
    ).start()
    logger.info(f"🩺 Health endpoint on http://{_host()}:{_port()}/healthz")
    return True


def stop() -> None:
    global _SERVER
    if _SERVER is not None:
        _SERVER.shutdown()
        _SERVER.server_close()
        _SERVER = None
