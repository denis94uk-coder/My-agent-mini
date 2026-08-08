"""
Phase 1.2 — governor: risk tiers, the approval queue, allow_risky.

Invariants under test:
  • Every registered tool has a tier, checked against the LIVE registry.
  • A tool nobody classified resolves to EXTERNAL (fail safe).
  • A parked approval survives a process restart and, once approved, runs the
    EXACT original call args off disk — not a re-derived call.
  • Silence is a deny: expiry resolves to a refusal the run must work around.
  • Run-spawning tools stay refused under allow_risky and under approval.

`xfail(strict=True)` marks a CONFIRMED DEFECT — see it fail with
`pytest --runxfail`.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_support import ScriptedAI, audit_env, run_id_of, tool_call  # noqa: F401

import governor
import runner
import tools


# ── tiers ──

def test_the_tier_guard_enumerates_the_live_registry(audit_env):
    """The existing guard must read tools.TOOLS, not a frozen copy — otherwise
    it passes forever while new tools go unclassified."""
    body = (Path(__file__).with_name("test_governor.py").read_text()
            .split("def test_every_registered_tool_has_a_tier")[1].split("\ndef ")[0])
    assert "set(tools.TOOLS)" in body, "tier guard does not read the live registry"
    assert not set(tools.TOOLS) - set(governor.TOOL_TIERS)


def test_a_tool_registered_at_runtime_with_no_tier_is_external(audit_env, monkeypatch):
    """Register a brand-new tool the way the decorator does, then check the
    default. This is the case the guard is protecting against."""
    monkeypatch.setitem(tools.TOOLS, "wipe_the_prod_database", {
        "name": "wipe_the_prod_database", "description": "x", "params": "",
        "func": lambda: "boom",
    })
    assert "wipe_the_prod_database" not in governor.TOOL_TIERS
    assert governor.tier_of("wipe_the_prod_database") == governor.EXTERNAL
    assert set(tools.TOOLS) - set(governor.TOOL_TIERS) == {"wipe_the_prod_database"}, (
        "the build-breaking guard would have caught this")


# ── the parked-approval lifecycle ──

def test_parked_approval_survives_restart_and_runs_the_exact_args(audit_env, monkeypatch):
    """park → (process dies) → recover → approve → the persisted args execute."""
    args = {"path": "deploy/prod.yaml", "content": "replicas: 9", "branch": "main"}
    monkeypatch.setattr(runner, "_CALL_AI",
                        ScriptedAI(tool_call("github_write_file", **args)))
    row = runner.enqueue_run("ship it", source="schedule", owner_user_id="U_OWNER",
                             unattended=True)
    run_id = run_id_of(row)
    runner.execute_run(runner.get_run(run_id))
    assert runner.get_run(run_id)["status"] == runner.AWAITING_APPROVAL

    # A parked run holds no worker, and crash recovery must not touch it.
    runner.recover_interrupted_runs()
    assert runner.get_run(run_id)["status"] == runner.AWAITING_APPROVAL

    pending = governor.pending_for_run(run_id)
    assert pending["args"] == args, "the queue stored different args than requested"

    executed = {}
    monkeypatch.setattr(tools, "run_tool",
                        lambda name, a: executed.update(name=name, args=a) or "ok")
    decided = governor.decide(pending["id"], True, decided_by="U_OWNER")
    assert runner.resume_after_decision(decided) is True
    monkeypatch.setattr(runner, "_CALL_AI", ScriptedAI("all done"))
    runner.execute_run(runner.get_run(run_id))

    assert executed["name"] == "github_write_file"
    assert {k: v for k, v in executed["args"].items() if not k.startswith("_")} == args


def test_approved_owner_only_tool_runs_as_the_decider_not_as_the_run(audit_env, monkeypatch):
    """Approval sanctions the ACTION; it does not promote the run to owner."""
    monkeypatch.setattr(runner, "_CALL_AI",
                        ScriptedAI(tool_call("push_branch", branch="x")))
    row = runner.enqueue_run("push it", source="schedule", owner_user_id="U_STRANGER",
                             unattended=True)
    run_id = run_id_of(row)
    runner.execute_run(runner.get_run(run_id))
    pending = governor.pending_for_run(run_id)
    seen = {}
    monkeypatch.setattr(tools, "run_tool", lambda n, a: seen.update(a) or "ok")
    decided = governor.decide(pending["id"], True, decided_by="U_OWNER")
    runner.resume_after_decision(decided)
    monkeypatch.setattr(runner, "_CALL_AI", ScriptedAI("done"))
    runner.execute_run(runner.get_run(run_id))
    assert seen.get("_requesting_user_id") == "U_OWNER"


def test_expiry_is_a_deny_and_the_run_resumes_with_the_refusal(audit_env):
    ap = governor.request_approval(1, "push_branch", {"site": "prod"})
    expired = governor.expire_stale_approvals(now=ap["expires_at"] + 1)
    assert [e["id"] for e in expired] == [ap["id"]]
    row = governor.get_approval(ap["id"])
    assert row["status"] == governor.EXPIRED
    assert "Treat this as a no" in governor.refusal_text(row)
    # An expired request can never be revived into consent.
    assert governor.decide(ap["id"], True, decided_by="U_OWNER") is None


def test_expired_decision_is_replayed_into_the_transcript_as_a_refusal(audit_env, monkeypatch):
    row = runner.enqueue_run("deploy", source="schedule", owner_user_id="U_OWNER",
                             unattended=True)
    run_id = run_id_of(row)
    ap = governor.request_approval(run_id, "push_branch", {"site": "prod"})
    governor.expire_stale_approvals(now=ap["expires_at"] + 1)
    messages = []
    runner._apply_pending_decision(runner.get_run(run_id), messages)
    assert messages and "Not approved" in messages[0]["content"]
    assert governor.get_approval(ap["id"])["executed"] == 1


@pytest.mark.parametrize("state", ["cancelled", "failed", "done"])
def test_approving_after_the_run_ended_does_not_execute(audit_env, monkeypatch, state):
    row = runner.enqueue_run("ship it", owner_user_id="U_OWNER", unattended=True)
    run_id = run_id_of(row)
    ap = governor.request_approval(run_id, "push_branch", {"branch": "x"})
    runner._update_run(run_id, status=state)
    decided = governor.decide(ap["id"], True, decided_by="U_OWNER")
    assert runner.resume_after_decision(decided) is False, (
        f"a {state} run was put back in the queue")
    ran = []
    monkeypatch.setattr(tools, "run_tool", lambda n, a: ran.append(n) or "ok")
    assert ran == []


def test_double_approval_and_bogus_ids_are_rejected(audit_env):
    ap = governor.request_approval(1, "push_branch", {"branch": "x"})
    assert governor.decide(ap["id"], True, decided_by="U_OWNER") is not None
    assert governor.decide(ap["id"], True, decided_by="U_OWNER") is None, "double-approve"
    assert governor.decide(ap["id"], False, decided_by="U_OWNER") is None, "deny-after-approve"
    assert governor.decide(999_999, True, decided_by="U_OWNER") is None, "bogus id"


# ── allow_risky ──

def test_run_spawning_tools_are_refused_even_with_allow_risky(audit_env):
    """allow_risky pre-authorises EXTERNAL effects, never the tools that queue
    more autonomous work — a run that can start runs can run away."""
    blocked = runner._blocked_tools_for({"unattended": True, "allow_risky": True})
    assert set(tools.UNATTENDED_BLOCKED_TOOLS) <= blocked
    gate = runner._approval_gate({"unattended": True, "allow_risky": True})
    for name in tools.UNATTENDED_BLOCKED_TOOLS:
        assert gate(name, {}) is True, f"{name} would be waved through by allow_risky"


def test_a_blocked_tool_never_reaches_the_approval_queue(audit_env, monkeypatch):
    """Blocked means blocked: it must not park the run on a request a human
    could grant."""
    monkeypatch.setattr(runner, "_CALL_AI",
                        ScriptedAI(tool_call("start_background_run", goal="spawn"), "done"))
    row = runner.enqueue_run("spawn more work", source="schedule",
                             owner_user_id="U_OWNER", unattended=True)
    run_id = run_id_of(row)
    runner.execute_run(runner.get_run(run_id))
    assert governor.pending_for_run(run_id) is None
    assert runner.get_run(run_id)["status"] != runner.AWAITING_APPROVAL


def test_allow_risky_is_per_schedule_and_has_no_user_facing_setter(audit_env):
    """INVARIANT: a non-owner cannot set allow_risky. Today nothing can: it is
    reachable only from triggers.add_schedule(allow_risky=True) in code — no
    slash command, no tool argument, no workflow preset."""
    import inspect
    import workflows
    assert "allow_risky" not in inspect.signature(tools.schedule_task).parameters
    assert "allow_risky" not in Path(workflows.__file__).read_text()
    default = runner.enqueue_run("plain", owner_user_id="U_OWNER", unattended=True)
    assert runner.get_run(run_id_of(default))["allow_risky"] in (0, False)


def test_approvals_disabled_turns_external_into_a_flat_no(audit_env, monkeypatch):
    monkeypatch.setenv("APPROVALS_ENABLED", "false")
    blocked = runner._blocked_tools_for({"unattended": True, "allow_risky": False})
    assert governor.external_tools() <= blocked
    assert runner._approval_gate({"unattended": True, "allow_risky": False}) is None


@pytest.mark.xfail(strict=True, reason=(
    "FINDING B1 (MEDIUM) — the approval decision has no authorisation check of "
    "its own. governor.decide (governor.py:286) accepts any decided_by string; "
    "the only owner check lives in bot.py:_decide_approval (bot.py:1177), i.e. "
    "in the Slack layer. Any other caller — a future HTTP hook, a tool, a "
    "REPL, a test-mode entry point — approves the owner's pending deploys. The "
    "module that owns the invariant does not enforce it."))
def test_decide_rejects_a_non_owner_caller(audit_env):
    ap = governor.request_approval(1, "push_branch", {"branch": "x"})
    assert governor.decide(ap["id"], True, decided_by="U_STRANGER") is None


@pytest.mark.xfail(strict=True, reason=(
    "FINDING B2 (LOW) — schedule_task tells the user something the engine does "
    "not do. tools.py:588 returns 'scheduled runs are unattended, so deploys, "
    "pushes and service restarts are blocked in them'. With APPROVALS_ENABLED "
    "(the default) those tools are NOT blocked: the run parks and asks, and "
    "with allow_risky it runs them outright. The message describes the "
    "approvals-off configuration only."))
def test_schedule_task_message_matches_actual_unattended_policy(audit_env):
    import inspect
    text = inspect.getsource(tools.schedule_task)
    external_blocked = governor.external_tools() <= runner._blocked_tools_for(
        {"unattended": True, "allow_risky": False})
    assert not ("blocked in them" in text and not external_blocked)
