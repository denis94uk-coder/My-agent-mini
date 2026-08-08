"""
Phase 2.6 — self-resuming plans (the sweeper).

Invariants under test:
  • Resume requires BOTH the plan and the Slack thread idle. A human mid-
    conversation is never talked over.
  • The per-conversation resume cap holds, and expires.
  • A plan that is waiting on a human is not resumed as if it were unblocked.

`xfail(strict=True)` marks a CONFIRMED DEFECT — see it fail with
`pytest --runxfail`.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_support import audit_env, run_id_of  # noqa: F401

import memory
import runner
import triggers


def _age_everything(conv_key, seconds, plan=True, messages=True):
    """Backdate the plan and/or the conversation so it looks idle."""
    conn = memory.get_db()
    try:
        if plan:
            conn.execute("UPDATE tasks SET updated = ? WHERE conv_key = ?",
                         (time.time() - seconds, conv_key))
        if messages:
            conn.execute("UPDATE conversations SET timestamp = ? WHERE conv_key = ?",
                         (time.time() - seconds, conv_key))
        conn.commit()
    finally:
        conn.close()


def test_sweeper_needs_both_plan_idle_and_thread_idle(audit_env):
    conv = "C1:1"
    memory.create_plan(conv, "U_OWNER", ["step one", "step two"])
    memory.add_message(conv, "user", "hang on, I'm still typing")
    now = time.time()

    _age_everything(conv, 4000, plan=True, messages=False)
    assert conv not in triggers.stale_plan_conv_keys(now=now, stale_seconds=600), (
        "the sweeper would have interrupted a live conversation")

    _age_everything(conv, 4000, plan=False, messages=True)
    assert conv in triggers.stale_plan_conv_keys(now=now, stale_seconds=600)


def test_a_finished_plan_is_never_swept(audit_env):
    conv = "C2:1"
    memory.create_plan(conv, "U_OWNER", ["only step"])
    memory.update_task_status(conv, 1, "done")
    _age_everything(conv, 4000)
    assert conv not in triggers.stale_plan_conv_keys(now=time.time(), stale_seconds=600)


def test_a_conversation_with_an_active_run_is_not_swept_again(audit_env, monkeypatch):
    """Otherwise every tick queues another continuation of the same plan."""
    conv = "C3:1"
    memory.create_plan(conv, "U_OWNER", ["step one"])
    _age_everything(conv, 4000)
    runner.enqueue_run("already running", source="plan_resume",
                       owner_user_id="U_OWNER", conv_key=conv)
    assert triggers._has_active_run(conv) is True
    assert triggers.sweep_stale_plans(now=time.time()) == []


def test_resume_cap_holds_per_conversation(audit_env, monkeypatch):
    monkeypatch.setenv("PLAN_MAX_RESUMES", "2")
    conv = "C4:1"
    memory.create_plan(conv, "U_OWNER", ["step one"])
    _age_everything(conv, 4000)

    queued = []
    for _ in range(4):
        ids = triggers.sweep_stale_plans(now=time.time())
        queued += ids
        # finish the run so the active-run guard doesn't mask the cap
        for rid in ids:
            runner._update_run(run_id_of(rid), status="done")
    assert len(queued) == 2, f"resume cap not enforced: {queued}"


def test_resume_cap_window_is_a_rolling_24h_not_a_calendar_day(audit_env, monkeypatch):
    """Documenting the actual semantics: triggers.py:509 counts plan_resume runs
    created in the last 86400s. There is no midnight reset — a run 24h+1s old
    stops counting, wherever the date boundary fell."""
    monkeypatch.setenv("PLAN_MAX_RESUMES", "1")
    conv = "C5:1"
    memory.create_plan(conv, "U_OWNER", ["step one"])
    _age_everything(conv, 4000)

    row = runner.enqueue_run("earlier resume", source="plan_resume",
                             owner_user_id="U_OWNER", conv_key=conv)
    rid = run_id_of(row)
    runner._update_run(rid, status="done")
    assert triggers._resume_count(conv, time.time() - 86400) == 1
    assert triggers.sweep_stale_plans(now=time.time()) == [], "cap should bite"

    # Age that run past the window; the cap releases without any date change.
    conn = memory.get_db()
    conn.execute("UPDATE runs SET created = ? WHERE id = ?",
                 (time.time() - 86401, rid))
    conn.commit()
    conn.close()
    assert triggers._resume_count(conv, time.time() - 86400) == 0
    assert triggers.sweep_stale_plans(now=time.time()), "cap should have released"


def test_auto_resume_can_be_switched_off(audit_env, monkeypatch):
    monkeypatch.setenv("PLAN_AUTO_RESUME", "false")
    conv = "C6:1"
    memory.create_plan(conv, "U_OWNER", ["step one"])
    _age_everything(conv, 4000)
    assert triggers.sweep_stale_plans(now=time.time()) == []


def test_resumed_plan_run_is_unattended_and_carries_the_plan_goal(audit_env):
    conv = "C7:1"
    memory.create_plan(conv, "U_OWNER", ["publish the report", "tell the team"])
    _age_everything(conv, 4000)
    ids = triggers.sweep_stale_plans(now=time.time())
    assert ids
    run = runner.get_run(run_id_of(ids[0]))
    assert run["unattended"] in (1, True)
    assert run["source"] == "plan_resume"
    assert "publish the report" in run["goal"]


def test_a_plan_waiting_on_a_human_is_not_resumed(audit_env):
    """FIXED (was FINDING D3): a step marked `blocked` takes the whole plan out
    of the sweeper's view until a human unblocks it."""
    conv = "C8:1"
    memory.create_plan(conv, "U_OWNER",
                       ["ask the owner which domain to deploy to",
                        "deploy once they answer"])
    conn = memory.get_db()
    conn.execute("UPDATE tasks SET status = 'blocked' WHERE conv_key = ? AND step_no = 1",
                 (conv,))
    conn.commit()
    conn.close()
    _age_everything(conv, 4000)
    assert triggers.stale_plan_conv_keys(now=time.time(), stale_seconds=600) == [], (
        "a plan waiting on a human answer was queued as unattended work")
