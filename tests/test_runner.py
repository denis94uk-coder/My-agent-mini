"""
Tests for the durable run engine.

Everything here runs against a temporary SQLite file with a fake AI, so the
suite needs no Slack workspace, no API keys, and no network.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memory
import runner
import tools


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point every module's storage at a throwaway database."""
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "memory.db")
    # The critic gate is exercised in test_critic.py. Off here so these tests
    # measure the run engine's own steps and events, not the extra AI call.
    monkeypatch.setenv("CRITIC_ENABLED", "false")
    monkeypatch.setattr(runner, "_CALL_AI", None)
    monkeypatch.setattr(runner, "_POST_MESSAGE", None)
    yield


class ScriptedAI:
    """Returns canned replies in order; records the prompts it was given."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, system_prompt):
        self.calls.append((list(messages), system_prompt))
        if not self.replies:
            return "Done — nothing left to do."
        return self.replies.pop(0)


def tool_call(name: str, **args) -> str:
    import json
    return f'[TOOL_CALL]\n{{"tool": "{name}", "args": {json.dumps(args)}}}\n[/TOOL_CALL]'


# ── Queueing and budgets ──

def test_enqueue_and_fetch():
    run_id = runner.enqueue_run("Do a thing", source="manual", owner_user_id="U1")
    run = runner.get_run(run_id)
    assert run["status"] == "queued"
    assert run["goal"] == "Do a thing"
    assert run["steps_used"] == 0


def test_empty_goal_rejected():
    with pytest.raises(runner.RunRejected):
        runner.enqueue_run("   ")


def test_daily_limit_only_applies_to_autonomous_runs(monkeypatch):
    monkeypatch.setenv("RUN_DAILY_LIMIT", "2")
    runner.enqueue_run("a", source="schedule")
    runner.enqueue_run("b", source="schedule")

    with pytest.raises(runner.RunRejected, match="Daily autonomous run limit"):
        runner.enqueue_run("c", source="plan_resume")

    # A human asking directly is never blocked by the autonomous cap.
    assert runner.enqueue_run("d", source="manual")


def test_cancel_marks_run_cancelled():
    run_id = runner.enqueue_run("cancel me")
    assert runner.cancel_run(run_id) is True
    assert runner.get_run(run_id)["status"] == "cancelled"
    # Already terminal — a second cancel changes nothing.
    assert runner.cancel_run(run_id) is False


# ── Execution ──

def test_run_executes_tools_then_finishes(monkeypatch):
    ai = ScriptedAI(
        tool_call("run_python", code="print(2 + 2)"),
        "The answer is 4.",
    )
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("Compute 2+2", owner_user_id="U1")
    final = runner.execute_run(runner.get_run(run_id))

    assert final["status"] == "done"
    assert "4" in final["result"]
    assert final["steps_used"] == 2

    kinds = [e["kind"] for e in runner.get_events(run_id)]
    assert kinds == ["assistant", "tool_result", "final"]


def test_step_budget_stops_the_run_and_still_reports(monkeypatch):
    # A model that never stops calling tools.
    ai = ScriptedAI(*[tool_call("run_python", code="print(1)")] * 10)
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("loop forever", max_steps=3)
    final = runner.execute_run(runner.get_run(run_id))

    assert final["status"] == "done"
    assert final["steps_used"] == 3
    assert "step budget" in final["result"]


def test_time_budget_stops_the_run(monkeypatch):
    ai = ScriptedAI(*[tool_call("run_python", code="print(1)")] * 5)
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("slow work", max_seconds=30)
    run = runner.get_run(run_id)
    # Pretend the clock already ran out.
    run["max_seconds"] = -1
    final = runner.execute_run(run)

    assert final["status"] == "done"
    assert "time budget" in final["result"]
    assert final["steps_used"] == 0


def test_cancelled_run_stops_before_calling_the_ai():
    ai = ScriptedAI("should never be reached")
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("cancel me")
    run = runner.get_run(run_id)
    runner.cancel_run(run_id)

    final = runner.execute_run(run)
    assert final["status"] == "cancelled"
    assert ai.calls == []


def test_ai_exception_queues_a_retry_with_backoff():
    def explode(messages, prompt):
        raise RuntimeError("provider on fire")

    runner.configure(call_ai_fn=explode, system_prompt="Test bot.")
    run_id = runner.enqueue_run("boom")
    final = runner.execute_run(runner.get_run(run_id))

    # A provider blip is transient — the run waits and tries again rather
    # than dying on the first error.
    assert final["status"] == "queued"
    assert final["attempts"] == 1
    assert final["next_attempt_at"] > time.time()
    assert "provider on fire" in final["error"]


def test_retries_are_not_claimable_until_their_backoff_expires():
    def explode(messages, prompt):
        raise RuntimeError("provider on fire")

    runner.configure(call_ai_fn=explode, system_prompt="Test bot.")
    run_id = runner.enqueue_run("boom")
    runner.execute_run(runner.get_run(run_id))

    assert runner._claim_next_run("w1") is None  # still backing off
    runner._update_run(run_id, next_attempt_at=time.time() - 1)
    assert runner._claim_next_run("w1")["id"] == run_id


def test_run_fails_for_good_once_retries_are_exhausted(monkeypatch):
    monkeypatch.setenv("RUN_MAX_RETRIES", "2")

    def explode(messages, prompt):
        raise RuntimeError("provider on fire")

    runner.configure(call_ai_fn=explode, system_prompt="Test bot.")
    run_id = runner.enqueue_run("boom")

    for _ in range(3):
        runner._update_run(run_id, status="running", next_attempt_at=None)
        final = runner.execute_run(runner.get_run(run_id))

    assert final["status"] == "failed"
    assert final["attempts"] == 3


def test_permanent_errors_are_not_retried():
    """A misconfiguration fails identically forever — retrying just burns quota."""
    runner.configure(system_prompt="Test bot.")
    runner._CALL_AI = None
    run_id = runner.enqueue_run("no backend")
    final = runner.execute_run(runner.get_run(run_id))

    assert final["status"] == "failed"
    assert final["attempts"] == 1


# ── Durability: the whole point of the engine ──

def test_interrupted_run_resumes_with_its_context():
    ai = ScriptedAI(tool_call("run_python", code="print('first half')"))
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("two-part job", max_steps=1)
    runner.execute_run(runner.get_run(run_id))  # burns its budget after one tool

    # Simulate the process dying mid-run and coming back up.
    runner._update_run(run_id, status="running")
    assert runner.recover_interrupted_runs() == 1

    resumed = runner.get_run(run_id)
    assert resumed["status"] == "queued"
    assert resumed["resume_count"] == 1

    messages = runner.rebuild_messages(resumed)
    # Goal + the assistant turn + the tool result it already saw.
    assert messages[0]["content"] == "two-part job"
    assert any(m["role"] == "assistant" for m in messages)
    assert any("TOOL_RESULT for run_python" in m["content"] for m in messages)


def test_repeatedly_interrupted_run_is_failed_not_retried_forever():
    run_id = runner.enqueue_run("cursed job")
    runner._update_run(run_id, status="running", resume_count=3)

    assert runner.recover_interrupted_runs() == 0
    assert runner.get_run(run_id)["status"] == "failed"


def test_claim_is_exclusive():
    run_id = runner.enqueue_run("only once")
    first = runner._claim_next_run("w1")
    second = runner._claim_next_run("w2")

    assert first["id"] == run_id
    assert second is None
    assert runner.get_run(run_id)["worker"] == "w1"


# ── Unattended safety envelope ──

def test_unattended_runs_block_state_changing_tools():
    run = {"unattended": 1, "allow_risky": 0}
    blocked = runner._blocked_tools_for(run)
    assert "deploy_static_site" in blocked
    assert "push_branch" in blocked
    assert "start_background_run" in blocked


def test_allow_risky_unlocks_owner_tools_but_never_run_spawning():
    run = {"unattended": 1, "allow_risky": 1}
    blocked = runner._blocked_tools_for(run)
    assert "deploy_static_site" not in blocked
    # Spawning more autonomous work stays blocked either way.
    assert blocked == set(tools.UNATTENDED_BLOCKED_TOOLS)


def test_attended_runs_block_nothing():
    assert runner._blocked_tools_for({"unattended": 0, "allow_risky": 0}) == set()


def test_blocked_tool_is_refused_but_the_run_continues():
    ai = ScriptedAI(
        tool_call("deploy_static_site", site_name="demo"),
        "I couldn't deploy — that needs a human.",
    )
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("ship the site", unattended=True)
    final = runner.execute_run(runner.get_run(run_id))

    assert final["status"] == "done"
    refusals = [e for e in runner.get_events(run_id) if e["kind"] == "tool_result"]
    assert "Blocked" in refusals[0]["content"]


# ── Reporting ──

def test_completion_is_written_to_conversation_memory():
    ai = ScriptedAI("All done: found three open PRs.")
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("check PRs", conv_key="C123:main", owner_user_id="U1")
    runner.execute_run(runner.get_run(run_id))

    history = memory.get_history("C123:main", limit=5)
    assert any("autonomous run" in m["content"] for m in history)
    assert any("three open PRs" in m["content"] for m in history)


def test_result_is_posted_to_the_originating_channel():
    posted = []
    ai = ScriptedAI("Weekly summary ready.")
    runner.configure(
        call_ai_fn=ai,
        system_prompt="Test bot.",
        post_message=lambda channel, thread, text: posted.append((channel, thread, text)),
    )

    run_id = runner.enqueue_run("summarize", channel="C999", thread_ts="123.45", label="weekly")
    runner.execute_run(runner.get_run(run_id))

    assert len(posted) == 1
    channel, thread, text = posted[0]
    assert (channel, thread) == ("C999", "123.45")
    assert "Weekly summary ready." in text


def test_conv_target_decodes_channel_and_thread():
    assert runner.conv_target("C123:1712.99") == ("C123", "1712.99")
    assert runner.conv_target("C123:main") == ("C123", "")
    assert runner.conv_target("") == ("", "")


def test_format_runs_lists_recent_runs():
    runner.enqueue_run("first job")
    runner.enqueue_run("second job")
    output = runner.format_runs()
    assert "first job" in output and "second job" in output


def test_unattended_prompt_tells_the_model_nobody_is_watching():
    ai = ScriptedAI("Reported.")
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("scheduled thing", unattended=True)
    runner.execute_run(runner.get_run(run_id))

    _, prompt = ai.calls[0]
    assert "UNATTENDED RUN" in prompt
    assert "do not ask clarifying questions" in prompt.lower()
