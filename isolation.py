"""
Process isolation for the child processes the agent spawns.

`run_shell` and `run_python` hand model-written code to the host. Until now the
only thing standing between that code and the box was `BLOCKED_PATTERNS`, a
regex denylist — and a denylist over a shell language with `$()`, aliases,
base64 and a hundred ways to spell the same command is not a security boundary.
It is a speed bump, and the tools' own comments say so.

This module adds the boundary the denylist was pretending to be, in two layers
that fail independently:

  1. **Environment scrubbing** (always on, no dependencies). The child gets an
     allowlisted environment with secret-shaped variables removed. This closes
     the single most likely leak: `echo $GITHUB_TOKEN` prints a live credential
     with no keyword anywhere near it, which is exactly the shape the output
     redaction in `tools.py` was added to catch *after* the fact. Removing the
     variable means there is nothing to redact.

  2. **Filesystem/process isolation via bubblewrap** (`bwrap`), when available.
     The root filesystem goes in read-only, a small set of paths are bound
     read-write, `/tmp` is a private tmpfs, and `.env` is masked with
     `/dev/null` so even a read-only root cannot surrender the tokens.

Layer 1 is pure Python and always applies. Layer 2 is best-effort: if `bwrap`
is not installed the command still runs, scrubbed but unconfined. That is a
deliberate choice — this box is currently a 1 GB VM where a hard dependency on
a package that may not be installed would take the agent's shell offline
entirely, which is a worse outcome than the status quo. `sandbox_status()`
reports which layers are actually in force so the degraded case is visible
rather than silent.

Set `SANDBOX_ENABLED=0` to disable layer 2 (layer 1 always applies).
"""

import os
import shutil
import functools
import logging

logger = logging.getLogger("my-agent-mini")


# ── Layer 1: environment scrubbing ──

# Variables the child genuinely needs to behave like a normal shell. Anything
# not in this list has to survive the secret-shape check below to be passed on.
_ENV_ALLOWLIST = {
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "TZ",
    "LANG", "LC_ALL", "LC_CTYPE", "PWD", "TMPDIR",
    "PYTHONPATH", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE",
    "VIRTUAL_ENV", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_RUNTIME_DIR",
}

# Name fragments that mark a variable as credential-bearing. Deliberately the
# same vocabulary as the output redaction in `tools.py` — if a name is worth
# redacting on the way out, it is worth withholding on the way in.
_SECRET_NAME_FRAGMENTS = (
    "TOKEN", "SECRET", "KEY", "PASSWORD", "PASSWD", "CREDENTIAL",
    "WEBHOOK", "SIGNING", "AUTH", "SESSION", "COOKIE", "PRIVATE",
)


def _is_secret_name(name: str) -> bool:
    upper = name.upper()
    return any(fragment in upper for fragment in _SECRET_NAME_FRAGMENTS)


def scrubbed_env(extra: dict | None = None) -> dict:
    """Build the environment a sandboxed child should see.

    Allowlisted names pass through as-is. Everything else is dropped if its
    name looks credential-bearing, and kept otherwise — a child that needs
    `EDITOR` or `GOPATH` still gets it, while `SLACK_BOT_TOKEN` never appears.
    """
    env = {}
    for name, value in os.environ.items():
        if name in _ENV_ALLOWLIST:
            env[name] = value
        elif not _is_secret_name(name):
            env[name] = value
    env.setdefault("HOME", os.path.expanduser("~"))
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    if extra:
        env.update(extra)
    return env


# ── Layer 2: bubblewrap ──

@functools.lru_cache(maxsize=1)
def bwrap_path() -> str | None:
    """Absolute path to `bwrap`, or None when it is not installed."""
    return shutil.which("bwrap")


def sandbox_requested() -> bool:
    """Whether layer 2 is switched on in config (independent of availability)."""
    return os.environ.get("SANDBOX_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def sandbox_active() -> bool:
    """Whether layer 2 will actually be applied to the next command."""
    return sandbox_requested() and bwrap_path() is not None


def sandbox_status() -> str:
    """One line describing which layers are in force, for `server_health`."""
    if not sandbox_requested():
        return "env-scrub only (SANDBOX_ENABLED=0; bwrap disabled by config)"
    if bwrap_path() is None:
        return "env-scrub only (bwrap not installed — `apt install bubblewrap`)"
    return "env-scrub + bwrap (read-only root, private /tmp, .env masked)"


def _masked_paths() -> list[str]:
    """Files to bind `/dev/null` over even though the root is read-only.

    A read-only root still *reads*, and `.env` is the one file on this box
    whose contents are equivalent to every credential the agent holds.
    """
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.expanduser("~/.env"),
        os.path.expanduser("~/.git-credentials"),
        os.path.expanduser("~/.netrc"),
        os.path.expanduser("~/.ssh"),
        os.path.expanduser("~/.aws"),
    ]
    return [p for p in candidates if os.path.exists(p)]


def wrap(
    argv: list[str],
    *,
    writable: list[str] | None = None,
    network: bool = True,
) -> list[str]:
    """Return `argv` wrapped in bubblewrap, or unchanged when unavailable.

    `writable` lists directories the child may modify — everything else on the
    filesystem is visible read-only. `network` defaults to True because the
    agent's shell is legitimately used for `git`, `pip` and `curl`; cutting the
    network by default would break the workflows this box exists to run.

    Callers must not assume the wrap happened. Pair this with `scrubbed_env`,
    which has no such dependency.
    """
    bwrap = bwrap_path()
    if not sandbox_requested() or bwrap is None:
        return list(argv)

    args = [
        bwrap,
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        # Die with the parent so a wedged child cannot outlive the run that
        # spawned it — the watchdog in `runner.py` fails such a run on a stale
        # heartbeat, and an orphaned process would keep working invisibly.
        "--die-with-parent",
        # A fresh session detaches the child from the bot's controlling
        # terminal, which otherwise allows TIOCSTI-style input injection back
        # into the parent.
        "--new-session",
        # No new privileges: setuid binaries reachable on the read-only root
        # cannot be used to escalate out.
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
    ]
    if not network:
        args += ["--unshare-net"]

    for path in writable or []:
        if os.path.isdir(path):
            args += ["--bind", path, path]

    for path in _masked_paths():
        # A directory cannot take a /dev/null bind; give it an empty tmpfs.
        if os.path.isdir(path):
            args += ["--tmpfs", path]
        else:
            args += ["--ro-bind", "/dev/null", path]

    return args + list(argv)
