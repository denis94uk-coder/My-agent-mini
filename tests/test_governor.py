"""
Tests for the governor: risk tiers, the approval queue, and cost accounting.

The properties that matter most: silence never becomes consent, an approval
survives a restart, and a tool nobody classified is treated as dangerous.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import governor
import memory
import runner
import tools


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "memory.db")
    monkeypatch.setattr(runner, "_CALL_AI", None)
    monkeypatch.setattr(runner, "_POST_MESSAGE", None)
    monkeypatch.setenv("CRITIC_ENABLED", "false")
    yield


def tool_call(name: str, **args) -> str:
    import json
    return f'[TOOL_CALL]\n{{"tool": "{name}", "args": {json.dumps(args)}}}\n[/TOOL_CALL]'


class ScriptedAI:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, system_prompt):
        self.calls.append((list(messages), system_prompt))
        return self.replies.pop(0) if self.replies else "Nothing left to do."


# ── Risk tiers ──

def test_every_registered_tool_has_a_tier():
    """
    Guards the fail-safe default. Unclassified tools are treated as EXTERNAL,
    which is right but noisy — this test means the default never fires in
    practice, because adding a tool without classifying it breaks the build.
    """
    unclassified = set(tools.TOOLS) - set(governor.TOOL_TIERS)
    assert not unclassified, f"tools missing a risk tier: {sorted(unclassified)}"


def test_tiers_do_not_name_tools_that_no_longer_exist():
    stale = set(governor.TOOL_TIERS) - set(tools.TOOLS)
    assert not stale, f"risk tiers name tools that don't exist: {sorted(stale)}"


def test_unknown_tools_are_treated_as_external():
    assert governor.tier_of("some_new_deploy_tool") == governor.EXTERNAL


@pytest.mark.parametrize("tool_name,tier", [
    ("web_search", governor.READ),
    ("read_file", governor.READ),
    ("run_shell", governor.WRITE_LOCAL),
    ("write_file", governor.WRITE_LOCAL),
    ("deploy_static_site", governor.EXTERNAL),
    ("push_branch", governor.EXTERNAL),
    ("restart_service", governor.EXTERNAL),
])
def test_representative_tiers(tool_name, tier):
    assert governor.tier_of(tool_name) == tier


def test_every_owner_only_tool_is_external():
    """Owner-only and externally-visible must not drift apart."""
    for name in tools.OWNER_ONLY_TOOLS:
        assert governor.tier_of(name) == governor.EXTERNAL, name


# ── Approval queue ──

def test_request_and_decide():
    approval = governor.request_approval(1, "deploy_static_site", {"site_name": "demo"})
    assert approval["status"] == governor.PENDING
    assert approval["args"] == {"site_name": "demo"}
    assert approval["tier"] == governor.EXTERNAL

    decided = governor.decide(approval["id"], True, decided_by="U1")
    assert decided["status"] == governor.APPROVED
    assert decided["decided_by"] == "U1"


def test_internal_args_are_not_stored():
    approval = governor.request_approval(1, "push_branch", {"repo": "x", "_requesting_user_id": "U9"})
    assert "_requesting_user_id" not in approval["args"]


def test_deciding_twice_does_nothing():
    approval = governor.request_approval(1, "push_branch", {})
    assert governor.decide(approval["id"], True, decided_by="U1")
    assert governor.decide(approval["id"], False, decided_by="U2") is None
    assert governor.get_approval(approval["id"])["status"] == governor.APPROVED


def test_expiry_is_a_deny_not_a_yes(monkeypatch):
    """The property that matters most: silence must never become consent."""
    monkeypatch.setenv("APPROVAL_TIMEOUT_SECONDS", "60")
    approval = governor.request_approval(1, "deploy_static_site", {})

    assert governor.expire_stale_approvals(time.time()) == []      # not yet
    expired = governor.expire_stale_approvals(time.time() + 120)

    assert len(expired) == 1
    assert governor.get_approval(approval["id"])["status"] == governor.EXPIRED
    assert "Treat this as a no" in governor.refusal_text(governor.get_approval(approval["id"]))


def test_denial_text_carries_the_reason_and_forbids_workarounds():
    approval = governor.request_approval(1, "restart_service", {})
    denied = governor.decide(approval["id"], False, decided_by="U1", note="not during business hours")
    text = governor.refusal_text(denied)
    assert "not during business hours" in text
    assert "another way" in text


def test_approval_request_message_shows_what_it_wants_to_do():
    run_id = runner.enqueue_run("ship the marketing site")
    approval = governor.request_approval(run_id, "deploy_static_site", {"site_name": "promo"})
    text = governor.format_approval_request(approval, runner.get_run(run_id))

    assert "deploy_static_site" in text
    assert "promo" in text
    assert "ship the marketing site" in text
    assert f"/approve {approval['id']}" in text
    assert "counts as a deny" in text


# ── Gate behaviour inside a run ──

def test_unattended_run_parks_on_an_external_tool():
    posted = []
    ai = ScriptedAI(tool_call("deploy_static_site", site_name="demo"))
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.",
                     post_message=lambda c, t, x: posted.append(x))

    run_id = runner.enqueue_run("deploy it", channel="C1", unattended=True)
    final = runner.execute_run(runner.get_run(run_id))

    assert final["status"] == runner.AWAITING_APPROVAL
    pending = governor.pending_for_run(run_id)
    assert pending["tool"] == "deploy_static_site"
    assert posted and "Approval needed" in posted[0]
    # The tool did NOT run, and no result was invented for it.
    assert not any(e["kind"] == "tool_result" for e in runner.get_events(run_id))


def test_parked_runs_are_not_claimable():
    ai = ScriptedAI(tool_call("push_branch", repo="x", branch_name="b", pr_title="t"))
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")
    run_id = runner.enqueue_run("push it", unattended=True)
    runner.execute_run(runner.get_run(run_id))

    assert runner._claim_next_run("w1") is None


def test_approved_tool_runs_on_resume_and_the_run_continues():
    ai = ScriptedAI(
        tool_call("write_file", filename="approved.txt", content="hello"),
        "Wrote the file.",
    )
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    # write_file is WRITE_LOCAL, so force the gate by pretending it's external.
    original = dict(governor.TOOL_TIERS)
    governor.TOOL_TIERS["write_file"] = governor.EXTERNAL
    try:
        run_id = runner.enqueue_run("write a file", unattended=True)
        runner.execute_run(runner.get_run(run_id))
        assert runner.get_run(run_id)["status"] == runner.AWAITING_APPROVAL

        approval = governor.pending_for_run(run_id)
        decided = governor.decide(approval["id"], True, decided_by="U1")
        assert runner.resume_after_decision(decided) is True
        assert runner.get_run(run_id)["status"] == "queued"

        governor.TOOL_TIERS.update(original)   # approval covers this one call
        final = runner.execute_run(runner.get_run(run_id))
    finally:
        governor.TOOL_TIERS.clear()
        governor.TOOL_TIERS.update(original)

    assert final["status"] == "done"
    results = [e for e in runner.get_events(run_id) if e["kind"] == "tool_result"]
    assert "Saved approved.txt" in results[0]["content"]
    assert governor.get_approval(approval["id"])["executed"] == 1


def test_denied_tool_becomes_a_refusal_the_agent_works_around():
    ai = ScriptedAI(
        tool_call("deploy_static_site", site_name="demo"),
        "Understood — I couldn't deploy, a human declined it.",
    )
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("deploy it", unattended=True)
    runner.execute_run(runner.get_run(run_id))

    approval = governor.pending_for_run(run_id)
    decided = governor.decide(approval["id"], False, decided_by="U1", note="wrong week")
    runner.resume_after_decision(decided)
    final = runner.execute_run(runner.get_run(run_id))

    assert final["status"] == "done"
    refusal = [e for e in runner.get_events(run_id) if e["kind"] == "tool_result"][0]
    assert "Denied" in refusal["content"]
    assert "wrong week" in refusal["content"]


def test_expired_approval_lets_the_run_finish_and_report():
    ai = ScriptedAI(
        tool_call("restart_service", service="my-agent"),
        "Nobody approved the restart, so it still needs doing by hand.",
    )
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("restart it", unattended=True)
    runner.execute_run(runner.get_run(run_id))

    for approval in governor.expire_stale_approvals(time.time() + 10**6):
        runner.resume_after_decision(approval)
    final = runner.execute_run(runner.get_run(run_id))

    assert final["status"] == "done"
    assert "still needs doing by hand" in final["result"]


def test_attended_runs_are_never_gated():
    """A human in the conversation is already the approval."""
    ai = ScriptedAI(tool_call("deploy_static_site", site_name="demo"), "Done.")
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("deploy it", unattended=False)
    final = runner.execute_run(runner.get_run(run_id))
    assert final["status"] == "done"
    assert governor.pending_for_run(run_id) is None


def test_allow_risky_skips_the_queue():
    """An owner who pre-authorised a schedule isn't asked again every night."""
    ai = ScriptedAI(tool_call("deploy_static_site", site_name="demo"), "Deployed.")
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("deploy it", unattended=True, allow_risky=True)
    final = runner.execute_run(runner.get_run(run_id))
    assert final["status"] == "done"
    assert governor.pending_for_run(run_id) is None


def test_run_spawning_tools_are_refused_not_queued():
    """Some things stay a flat no — asking can't make a fork bomb safe."""
    ai = ScriptedAI(tool_call("start_background_run", goal="another run"), "Couldn't.")
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("spawn more work", unattended=True)
    final = runner.execute_run(runner.get_run(run_id))

    assert final["status"] == "done"
    assert governor.pending_for_run(run_id) is None
    assert "Blocked" in [e for e in runner.get_events(run_id) if e["kind"] == "tool_result"][0]["content"]


def test_approvals_can_be_switched_off_entirely(monkeypatch):
    monkeypatch.setenv("APPROVALS_ENABLED", "false")
    ai = ScriptedAI(tool_call("deploy_static_site", site_name="demo"), "Blocked, reported.")
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("deploy it", unattended=True)
    final = runner.execute_run(runner.get_run(run_id))

    assert final["status"] == "done"          # hard block, no queue
    assert governor.pending_for_run(run_id) is None


def test_parked_run_survives_a_restart():
    """The approval is on disk, so a restart doesn't lose the request."""
    ai = ScriptedAI(tool_call("deploy_static_site", site_name="demo"))
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")
    run_id = runner.enqueue_run("deploy it", unattended=True)
    runner.execute_run(runner.get_run(run_id))

    # A parked run is not "running", so recovery must leave it alone.
    assert runner.recover_interrupted_runs() == 0
    assert runner.get_run(run_id)["status"] == runner.AWAITING_APPROVAL
    assert governor.pending_for_run(run_id)["tool"] == "deploy_static_site"


# ── Cost accounting ──

def test_calls_are_counted_per_provider():
    governor.record_ai_call("Pollinations", input_chars=100, output_chars=50)
    governor.record_ai_call("Pollinations", input_chars=200, output_chars=80)

    summary = {r["provider"]: r for r in governor.usage_summary()}
    assert summary["Pollinations"]["calls"] == 2
    assert summary["Pollinations"]["input_chars"] == 300
    assert summary["Pollinations"]["output_chars"] == 130


def test_paid_routes_are_tracked_separately(monkeypatch):
    monkeypatch.setenv("PAID_PROVIDERS", "merge")
    governor.record_ai_call("Merge Gateway")
    governor.record_ai_call("Pollinations")

    assert governor.paid_calls_today() == 1
    assert governor.is_paid("Merge Gateway") is True
    assert governor.is_paid("Pollinations") is False


def test_paid_budget_cap(monkeypatch):
    monkeypatch.setenv("PAID_PROVIDERS", "merge")
    monkeypatch.setenv("PAID_DAILY_LIMIT", "2")

    assert governor.paid_budget_exhausted() is False
    governor.record_ai_call("Merge Gateway")
    governor.record_ai_call("Merge Gateway")
    assert governor.paid_budget_exhausted() is True


def test_paid_cap_of_zero_means_uncapped(monkeypatch):
    monkeypatch.setenv("PAID_DAILY_LIMIT", "0")
    monkeypatch.setenv("PAID_PROVIDERS", "merge")
    governor.record_ai_call("Merge Gateway")
    assert governor.paid_budget_exhausted() is False


def test_accounting_never_raises(monkeypatch):
    """A broken counter must not break a reply."""
    monkeypatch.setattr(memory, "DB_PATH", Path("/nonexistent/dir/memory.db"))
    governor.record_ai_call("Pollinations")   # must not raise


def test_usage_summary_renders_for_slack(monkeypatch):
    monkeypatch.setenv("PAID_PROVIDERS", "merge")
    governor.record_ai_call("Merge Gateway", 10, 10)
    text = governor.format_usage()
    assert "Merge Gateway" in text
    assert "paid" in text


# ── Router ordering (cost protection) ──

def test_paid_routes_sort_last(monkeypatch):
    """
    Registration order is just the order the provider blocks are written, which
    had NVIDIA's free tier sitting behind the paid gateway — a Pollinations
    failure would have spent credit while a free route went untried.
    """
    monkeypatch.setenv("PAID_PROVIDERS", "merge")
    providers = [
        {"name": "Pollinations (keyless)"},
        {"name": "Groq"},
        {"name": "Merge Gateway"},
        {"name": "NVIDIA"},
    ]
    providers.sort(key=lambda p: 1 if governor.is_paid(p["name"]) else 0)

    names = [p["name"] for p in providers]
    assert names[-1] == "Merge Gateway"
    assert names.index("NVIDIA") < names.index("Merge Gateway")
    # A stable sort must leave the free ordering (keyless first) untouched.
    assert names[:3] == ["Pollinations (keyless)", "Groq", "NVIDIA"]


def test_multiple_paid_routes_can_be_named(monkeypatch):
    monkeypatch.setenv("PAID_PROVIDERS", "merge,openrouter")
    assert governor.is_paid("Merge Gateway")
    assert governor.is_paid("OpenRouter")
    assert not governor.is_paid("Groq")


def test_no_paid_providers_configured_means_nothing_is_paid(monkeypatch):
    monkeypatch.setenv("PAID_PROVIDERS", "")
    assert not governor.is_paid("Merge Gateway")
    assert governor.paid_budget_exhausted() is False
