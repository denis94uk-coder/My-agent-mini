"""
Child-process isolation for run_shell / run_python.

Two layers, tested separately because they fail separately:

  • Layer 1, environment scrubbing, is pure Python and always applies. Its
    invariant is that a secret-named variable never reaches the child, so
    `echo $GITHUB_TOKEN` has nothing to print. These tests must hold on every
    machine.

  • Layer 2, bubblewrap, is best-effort. Where a test needs `bwrap` on the
    box it is skipped rather than failed — an environment without it is a
    known degraded mode, not a regression. The one thing asserted
    unconditionally is that the degradation is *visible*.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import isolation

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap not installed; layer 2 is best-effort by design",
)


@pytest.fixture(autouse=True)
def _clear_bwrap_cache():
    """`bwrap_path` is lru_cached, and tests move PATH around underneath it."""
    isolation.bwrap_path.cache_clear()
    yield
    isolation.bwrap_path.cache_clear()


# ── Layer 1: environment scrubbing ──

def test_secret_named_variables_are_withheld(monkeypatch):
    """INVARIANT: the child cannot print what it was never given. This is the
    leak the output redaction in tools.py exists to catch after the fact."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_" + "A" * 36)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-nope")
    monkeypatch.setenv("SOME_API_KEY", "sk-nope")
    monkeypatch.setenv("STRIPE_SECRET", "nope")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "nope")

    env = isolation.scrubbed_env()

    for name in (
        "GITHUB_TOKEN", "SLACK_BOT_TOKEN", "SOME_API_KEY",
        "STRIPE_SECRET", "SLACK_SIGNING_SECRET",
    ):
        assert name not in env, f"{name} reached the sandboxed child"


def test_ordinary_variables_still_pass_through(monkeypatch):
    """Scrubbing must not be so broad that normal tooling breaks — a shell
    with no PATH or a git with no HOME is a tool the agent cannot use."""
    monkeypatch.setenv("EDITOR", "vim")
    monkeypatch.setenv("GOPATH", "/go")

    env = isolation.scrubbed_env()

    assert env["EDITOR"] == "vim"
    assert env["GOPATH"] == "/go"
    assert env.get("PATH")
    assert env.get("HOME")


def test_scrubbed_env_has_path_and_home_even_when_unset(monkeypatch):
    monkeypatch.delenv("PATH", raising=False)
    monkeypatch.delenv("HOME", raising=False)

    env = isolation.scrubbed_env()

    assert env["PATH"]
    assert env["HOME"]


def test_extra_overrides_win():
    env = isolation.scrubbed_env({"PATH": "/only/here"})
    assert env["PATH"] == "/only/here"


# ── Layer 2: bubblewrap wrapping ──

def test_wrap_is_a_passthrough_when_bwrap_is_missing(monkeypatch):
    """INVARIANT: a missing sandbox degrades to an unconfined command, it does
    not take the agent's shell offline. Layer 1 still applies either way."""
    monkeypatch.setattr(isolation.shutil, "which", lambda _: None)
    isolation.bwrap_path.cache_clear()

    assert isolation.wrap(["echo", "hi"]) == ["echo", "hi"]
    assert isolation.sandbox_active() is False


def test_wrap_is_a_passthrough_when_disabled_by_config(monkeypatch):
    monkeypatch.setenv("SANDBOX_ENABLED", "0")
    assert isolation.wrap(["echo", "hi"]) == ["echo", "hi"]
    assert isolation.sandbox_active() is False


def test_status_names_the_degraded_mode(monkeypatch):
    """A sandbox that quietly stopped applying is the same as no sandbox. The
    status string is the only thing that makes the difference observable."""
    monkeypatch.setattr(isolation.shutil, "which", lambda _: None)
    isolation.bwrap_path.cache_clear()
    assert "bwrap not installed" in isolation.sandbox_status()

    monkeypatch.setenv("SANDBOX_ENABLED", "0")
    assert "SANDBOX_ENABLED=0" in isolation.sandbox_status()


@needs_bwrap
def test_wrap_prefixes_bwrap_and_preserves_the_command():
    argv = isolation.wrap(["echo", "hi"])

    assert argv[0].endswith("bwrap")
    assert argv[-2:] == ["echo", "hi"]
    assert "--ro-bind" in argv
    assert "--die-with-parent" in argv


@needs_bwrap
def test_network_is_shared_by_default_and_droppable():
    """The agent's shell legitimately runs git and pip, so the default has to
    keep the network. Callers that want it gone must be able to say so."""
    assert "--unshare-net" not in isolation.wrap(["true"])
    assert "--unshare-net" in isolation.wrap(["true"], network=False)


@needs_bwrap
def test_root_filesystem_is_read_only(tmp_path):
    argv = isolation.wrap(["touch", "/etc/pwned-by-test"])
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)

    assert result.returncode != 0
    assert "read-only" in result.stderr.lower()
    assert not os.path.exists("/etc/pwned-by-test")


@needs_bwrap
def test_writable_paths_are_actually_writable(tmp_path):
    target = tmp_path / "written"
    argv = isolation.wrap(["touch", str(target)], writable=[str(tmp_path)])

    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)

    assert result.returncode == 0, result.stderr
    assert target.exists()


@needs_bwrap
def test_dotenv_is_masked_even_though_the_root_is_readable(monkeypatch, tmp_path):
    """INVARIANT: read-only still means readable, and .env is the one file whose
    contents are equivalent to every credential this box holds."""
    secret_env = tmp_path / ".env"
    secret_env.write_text("SLACK_BOT_TOKEN=xoxb-the-real-one\n")
    monkeypatch.setattr(isolation, "_masked_paths", lambda: [str(secret_env)])

    argv = isolation.wrap(["cat", str(secret_env)])
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)

    assert "xoxb-the-real-one" not in result.stdout
    assert "xoxb-the-real-one" not in result.stderr


@needs_bwrap
def test_a_masked_directory_becomes_empty_rather_than_failing(monkeypatch, tmp_path):
    """~/.ssh is a directory, and /dev/null cannot be bound over one. Getting
    this wrong makes bwrap refuse to start at all, which fails open."""
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "id_rsa").write_text("PRIVATE KEY MATERIAL")
    monkeypatch.setattr(isolation, "_masked_paths", lambda: [str(ssh_dir)])

    argv = isolation.wrap(["ls", str(ssh_dir)])
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)

    assert result.returncode == 0, result.stderr
    assert "id_rsa" not in result.stdout


# ── The tools themselves ──

def test_run_shell_cannot_echo_a_token(monkeypatch):
    """End to end through the real tool, on any machine: with or without bwrap,
    layer 1 alone is enough to make this impossible."""
    import tools

    monkeypatch.setenv("OWNER_SLACK_ID", "U_OWNER")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_" + "Z" * 36)

    out = tools.run_shell("echo [$GITHUB_TOKEN]")

    assert "ghp_" not in out
    assert "Z" * 36 not in out


def test_run_python_cannot_read_a_token_from_the_environment(monkeypatch):
    import tools

    monkeypatch.setenv("OWNER_SLACK_ID", "U_OWNER")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_" + "Y" * 36)

    out = tools.run_python(
        "import os; print('value:', os.environ.get('GITHUB_TOKEN'))"
    )

    assert "None" in out
    assert "Y" * 36 not in out


def test_run_python_still_works(monkeypatch):
    """The sandbox moved run_python's scratch file off /tmp, because /tmp is a
    private tmpfs inside the namespace. If that move were wrong, every call
    would fail with 'No such file or directory' instead of running."""
    import tools

    monkeypatch.setenv("OWNER_SLACK_ID", "U_OWNER")

    assert "4" in tools.run_python("print(2 + 2)")


def test_run_python_cleans_up_after_a_timeout(monkeypatch):
    """The unlink used to sit on the success path only, so every timed-out
    script leaked a file into the scratch directory forever."""
    import tools

    monkeypatch.setenv("OWNER_SLACK_ID", "U_OWNER")
    before = set(os.listdir(tools._PY_SCRATCH))

    monkeypatch.setattr(
        tools.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(cmd="python3", timeout=30)
        ),
    )
    out = tools.run_python("import time; time.sleep(99)")

    assert "timed out" in out
    assert set(os.listdir(tools._PY_SCRATCH)) == before


def test_run_shell_still_works(monkeypatch):
    import tools

    monkeypatch.setenv("OWNER_SLACK_ID", "U_OWNER")

    assert "hello" in tools.run_shell("echo hello")


def test_run_shell_can_write_to_the_workspace(monkeypatch):
    """The agent's whole file-handling surface lives under WORKSPACE. A sandbox
    that made it read-only would be secure and useless."""
    import tools

    monkeypatch.setenv("OWNER_SLACK_ID", "U_OWNER")

    out = tools.run_shell("touch sandbox_probe && ls sandbox_probe")

    assert "sandbox_probe" in out
    try:
        os.unlink(os.path.join(tools.WORKSPACE, "sandbox_probe"))
    except OSError:
        pass


def test_server_health_reports_the_sandbox_posture(monkeypatch):
    import tools

    monkeypatch.setenv("OWNER_SLACK_ID", "U_OWNER")

    assert "--- sandbox ---" in tools.server_health()
