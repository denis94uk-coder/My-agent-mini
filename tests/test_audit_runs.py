"""
Phase 2.5 — the run engine: durability, budgets, cancellation, concurrency.

Invariants under test:
  • A run that dies mid-step resumes AT that step with its transcript intact.
  • A run that stops heartbeating is failed, and its schedule is unblocked.
  • Hitting a budget reports partial work; it never drops the work silently.
  • The autonomy cap applies to self-started runs, never to a human's request.
  • Two workers never claim the same run, and concurrent writers never hit
    'database is locked'.

`xfail(strict=True)` marks a CONFIRMED DEFECT — see it fail with
`pytest --runxfail`.
"""

import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_support import ScriptedAI, audit_env, run_id_of, tool_call  # noqa: F401

import memory
import runner
import tools

REPO = str(Path(__file__).resolve().parent.parent)


# ── durability across a real SIGKILL ──

CHILD = '''
import sys, time
sys.path.insert(0, {repo!r})
from pathlib import Path
import memory, runner, tools

memory.DB_PATH = Path({db!r})

replies = [
    '[TOOL_CALL]\\n{{"tool": "list_files", "args": {{}}}}\\n[/TOOL_CALL]',
    '[TOOL_CALL]\\n{{"tool": "read_file", "args": {{"filename": "slow.txt"}}}}\\n[/TOOL_CALL]',
    "done",
]

def ai(messages, prompt=None):
    return replies.pop(0) if replies else "done"

tools.TOOLS["list_files"]["func"] = lambda: "STEP-ONE-RESULT"
tools.TOOLS["read_file"]["func"] = lambda filename: time.sleep(300) or "never"

runner.configure(call_ai_fn=ai, system_prompt="s")
row = runner.enqueue_run("two step job", owner_user_id="U_OWNER", unattended=True)
run_id = row["id"] if isinstance(row, dict) else row
runner._update_run(run_id, status="running", worker="child", heartbeat=time.time())
runner.execute_run(runner.get_run(run_id))
'''


def test_sigkill_mid_step_resumes_at_that_step_with_context(audit_env, monkeypatch):
    """kill -9 while a tool is executing: the finished step must survive on
    disk and the rebuilt transcript must still contain it."""
    monkeypatch.setenv("CRITIC_ENABLED", "false")
    db = str(memory.DB_PATH)
    script = Path(audit_env) / "child.py"
    script.write_text(CHILD.format(repo=REPO, db=db))

    env = dict(os.environ, OWNER_SLACK_ID="U_OWNER", CRITIC_ENABLED="false",
               PYTHONPATH=REPO)
    child = subprocess.Popen([sys.executable, str(script)], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            runs = runner.list_runs(limit=5)
            if runs and len(runner.get_events(runs[0]["id"])) >= 2:
                break
            time.sleep(0.1)
        else:
            child.kill()
            pytest.fail(f"child never reached step 2: {child.stderr.read()[:400]}")
        run_id = runner.list_runs(limit=5)[0]["id"]
        os.kill(child.pid, signal.SIGKILL)          # not SIGTERM: no cleanup runs
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()

    assert child.returncode == -signal.SIGKILL
    row = runner.get_run(run_id)
    assert row["status"] == "running", "the row should still look mid-flight"

    # Recovery: the interrupted run goes back in the queue, once.
    assert runner.recover_interrupted_runs() == 1
    row = runner.get_run(run_id)
    assert row["status"] == "queued"
    assert row["resume_count"] == 1, "resume_count must record the crash"
    assert row["attempts"] == 0, "a crash is not a retry — attempts must stay 0"

    replayed = "\n".join(str(m) for m in runner.rebuild_messages(row))
    assert "STEP-ONE-RESULT" in replayed, "resume restarted from zero"


def test_recovery_leaves_a_parked_run_alone(audit_env):
    """recover_interrupted_runs touches `running` only — a run awaiting a human
    holds no worker and must not be re-queued out from under the approval."""
    row = runner.enqueue_run("parked", owner_user_id="U_OWNER", unattended=True)
    run_id = run_id_of(row)
    runner._update_run(run_id, status=runner.AWAITING_APPROVAL)
    assert runner.recover_interrupted_runs() == 0
    assert runner.get_run(run_id)["status"] == runner.AWAITING_APPROVAL


# ── heartbeat watchdog ──

@pytest.mark.xfail(strict=True, reason=(
    "FINDING D2 (HIGH) — the watchdog does not achieve the one thing it exists "
    "for. Because sweep_stuck_runs re-queues (see FINDING D1), the row lands in "
    "status='queued', which is inside runner.ACTIVE_STATUSES, so "
    "triggers._schedule_run_active (triggers.py:369) still reports the schedule "
    "as running and the overlap guard (triggers.py:326) keeps skipping it. Its "
    "docstring promises 'the schedule behind it is unblocked'; a hung "
    "ops-watch/repo-review schedule therefore stays dead until someone "
    "restarts the process."))
def test_stuck_run_unblocks_its_schedule(audit_env, monkeypatch):
    """The row must stop counting as active, or the overlap guard would refuse
    to ever fire that schedule again."""
    import triggers
    monkeypatch.setenv("RUN_STUCK_SECONDS", "60")
    triggers.add_schedule(name="nightly", spec="every 1h", goal="watch",
                          owner_user_id="U_OWNER")
    row = runner.enqueue_run("watch", source="schedule", owner_user_id="U_OWNER",
                             unattended=True, label="nightly")
    run_id = run_id_of(row)
    runner._update_run(run_id, status="running", worker="w1",
                       heartbeat=time.time() - 3600)
    assert triggers._schedule_run_active("nightly") is True
    runner.sweep_stuck_runs()
    assert triggers._schedule_run_active("nightly") is False, (
        "the schedule is still blocked by the stuck run")


@pytest.mark.xfail(strict=True, reason=(
    "FINDING D1 (HIGH) — a hung run is re-queued. sweep_stuck_runs "
    "(runner.py:1039) hands the stall to _fail_or_retry (runner.py:644), whose "
    "error text is not in _PERMANENT_ERROR_MARKERS, so the run goes back to "
    "status='queued' with a 60s backoff. Both its own docstring "
    "('This does NOT re-queue the run') and CLAUDE.md ('it must never re-queue "
    "it, since the worker thread may still be live inside that tool') say the "
    "opposite. A second worker then executes the same unattended steps while "
    "the first is potentially still inside the tool — duplicate deploys, "
    "duplicate PRs, duplicate GitHub writes."))
def test_stuck_run_is_never_requeued(audit_env, monkeypatch):
    monkeypatch.setenv("RUN_STUCK_SECONDS", "60")
    row = runner.enqueue_run("hang forever", owner_user_id="U_OWNER", unattended=True)
    run_id = run_id_of(row)
    runner._update_run(run_id, status="running", worker="w1",
                       heartbeat=time.time() - 3600)
    runner.sweep_stuck_runs()
    assert runner.get_run(run_id)["status"] == "failed", (
        f"stuck run left in state {runner.get_run(run_id)['status']!r}")


# ── budgets ──

def test_step_budget_reports_partial_work(audit_env, monkeypatch):
    monkeypatch.setattr(runner, "_CALL_AI", ScriptedAI(*[tool_call("list_files")] * 8))
    row = runner.enqueue_run("loop forever", owner_user_id="U_OWNER",
                             unattended=True, max_steps=2)
    final = runner.execute_run(runner.get_run(run_id_of(row)))
    assert final["status"] == "done"
    assert "step budget" in final["result"].lower(), final["result"][:200]


def test_wall_clock_budget_reports_partial_work(audit_env, monkeypatch):
    def slow_ai(messages, prompt=None):
        time.sleep(0.15)
        return tool_call("list_files")
    monkeypatch.setattr(runner, "_CALL_AI", slow_ai)
    row = runner.enqueue_run("long job", owner_user_id="U_OWNER",
                             unattended=True, max_seconds=1)
    final = runner.execute_run(runner.get_run(run_id_of(row)))
    assert final["status"] == "done"
    assert "budget" in final["result"].lower()
    assert final["result"].strip(), "wall-clock stop produced nothing"


def test_daily_limit_blocks_self_started_runs_only(audit_env, monkeypatch):
    monkeypatch.setenv("RUN_DAILY_LIMIT", "1")
    runner.enqueue_run("auto 1", source="schedule", owner_user_id="U_OWNER")
    with pytest.raises(runner.RunRejected):
        runner.enqueue_run("auto 2", source="schedule", owner_user_id="U_OWNER")
    with pytest.raises(runner.RunRejected):
        runner.enqueue_run("auto 3", source="plan_resume", owner_user_id="U_OWNER")
    # A human asking is never blocked by the autonomy cap.
    runner.enqueue_run("human asked", source="manual", owner_user_id="U_OWNER")


def test_config_errors_never_retry(audit_env):
    """A misconfiguration fails identically forever; retrying burns quota."""
    for permanent in ("❌ Not authorized: run_shell", "No AI backend configured",
                      "Unknown tool: frobnicate"):
        assert runner._is_retryable(permanent) is False, permanent
    assert runner._is_retryable("connection reset by peer") is True


def test_retry_and_resume_are_counted_separately(audit_env):
    row = runner.enqueue_run("flaky", owner_user_id="U_OWNER", unattended=True)
    run_id = run_id_of(row)
    runner._fail_or_retry(runner.get_run(run_id), "provider timeout")
    after_retry = runner.get_run(run_id)
    assert after_retry["attempts"] == 1 and after_retry["resume_count"] == 0

    runner._update_run(run_id, status="running", heartbeat=0, worker="dead")
    runner.recover_interrupted_runs()
    after_resume = runner.get_run(run_id)
    assert after_resume["attempts"] == 1, "a crash inflated the retry counter"
    assert after_resume["resume_count"] == 1


# ── cancellation ──

def test_cancel_stops_at_the_next_step_boundary(audit_env, monkeypatch):
    """Cancel lands between steps: the in-flight tool completes, nothing new
    starts, and the transcript is not left half-written."""
    calls = []

    def ai(messages, prompt=None):
        calls.append(1)
        if len(calls) == 1:
            return tool_call("list_files")
        return tool_call("list_files")

    monkeypatch.setattr(runner, "_CALL_AI", ai)
    row = runner.enqueue_run("keep going", owner_user_id="U_OWNER", unattended=True)
    run_id = run_id_of(row)

    original = tools.run_tool

    def cancelling_tool(name, args):
        runner.cancel_run(run_id)          # cancelled *while* the tool runs
        return original(name, args)

    monkeypatch.setattr(tools, "run_tool", cancelling_tool)
    final = runner.execute_run(runner.get_run(run_id))

    assert final["status"] == "cancelled"
    assert len(calls) == 1, "a new step started after cancellation"
    kinds = [e["kind"] for e in runner.get_events(run_id)]
    assert kinds.count("assistant") == kinds.count("tool_result"), (
        f"half-written step left in the transcript: {kinds}")


def test_cancelling_a_finished_run_is_a_no_op(audit_env):
    row = runner.enqueue_run("done already", owner_user_id="U_OWNER")
    run_id = run_id_of(row)
    runner._update_run(run_id, status="done")
    assert runner.cancel_run(run_id) is False
    assert runner.get_run(run_id)["status"] == "done"


# ── concurrency ──

def test_two_workers_never_claim_the_same_run(audit_env):
    """The claim must be atomic: 2 workers, 3 queued runs, no double-claim."""
    for i in range(3):
        runner.enqueue_run(f"job {i}", owner_user_id="U_OWNER", unattended=True)

    claimed, errors = [], []
    lock = threading.Lock()

    def worker(name):
        try:
            while True:
                run = runner._claim_next_run(name)
                if not run:
                    return
                with lock:
                    claimed.append(run["id"])
        except Exception as e:  # pragma: no cover
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert not errors, errors
    assert sorted(claimed) == sorted(set(claimed)), f"a run was claimed twice: {claimed}"
    assert len(claimed) == 3


def test_concurrent_writers_do_not_hit_database_is_locked(audit_env):
    """Workers + scheduler + sweeper share one SQLite file. None may fail."""
    errors = []

    def writer(n):
        try:
            for i in range(40):
                memory.add_message(f"C{n}:1", "user", f"m{i}")
                row = runner.enqueue_run(f"run {n}-{i}", owner_user_id="U_OWNER")
                runner.add_event(run_id_of(row), "assistant", "x" * 200)
        except Exception as e:  # pragma: no cover
            errors.append(repr(e))

    def sweeper():
        try:
            for _ in range(40):
                runner.sweep_stuck_runs()
                runner.list_runs(limit=20)
        except Exception as e:  # pragma: no cover
            errors.append(repr(e))

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(3)]
    threads.append(threading.Thread(target=sweeper))
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors
