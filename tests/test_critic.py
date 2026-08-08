"""
Tests for the critic gate — the pass that decides whether "done" is done.

Two properties matter more than the happy path and are tested hardest:
the gate fails *open* (a broken critic never traps finished work), and it
never loops on work the run was structurally blocked from doing.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent
import critic
import memory
import runner


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "memory.db")
    # The owner lock fails closed; these tests exercise execution, not auth.
    monkeypatch.setenv("OWNER_SLACK_ID", "default")
    monkeypatch.setattr(runner, "_CALL_AI", None)
    monkeypatch.setattr(runner, "_POST_MESSAGE", None)
    yield


def tool_call(name: str, **args) -> str:
    import json
    return f'[TOOL_CALL]\n{{"tool": "{name}", "args": {json.dumps(args)}}}\n[/TOOL_CALL]'


class ScriptedAI:
    """Canned replies in order, recording every prompt it was handed."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, system_prompt):
        self.calls.append((list(messages), system_prompt))
        return self.replies.pop(0) if self.replies else "Nothing left to do."


# ── Verdict parsing ──

def test_accept_is_parsed():
    assert critic.parse_verdict("VERDICT: ACCEPT").accepted


def test_revise_carries_its_reason():
    verdict = critic.parse_verdict(
        "VERDICT: REVISE\nREASON: You said the file was saved but no write_file "
        "call appears in the transcript."
    )
    assert verdict.kind == critic.REVISE
    assert "write_file" in verdict.reason


def test_verdict_parsing_is_case_and_noise_tolerant():
    verdict = critic.parse_verdict("Thinking...\n\nverdict: revise\nreason: step 3 was skipped.")
    assert verdict.kind == critic.REVISE
    assert "step 3" in verdict.reason


@pytest.mark.parametrize("text", ["", "   ", "I think it looks good honestly", "REVISE"])
def test_unparseable_verdicts_fail_open(text):
    """A grader that has gone off the rails must not block finished work."""
    assert critic.parse_verdict(text).accepted


def test_revise_without_a_reason_fails_open():
    assert critic.parse_verdict("VERDICT: REVISE").accepted


def test_review_survives_a_crashing_ai():
    def explode(messages, prompt):
        raise RuntimeError("provider down")

    assert critic.review("goal", [], "my answer", explode).accepted


def test_review_treats_a_router_error_as_no_verdict():
    assert critic.review("goal", [], "answer", lambda m, p: "❌ All routes failed").accepted


def test_empty_answer_is_not_reviewed():
    calls = []
    critic.review("goal", [], "", lambda m, p: calls.append(1) or "VERDICT: REVISE\nREASON: x")
    assert calls == []


# ── Prompt construction ──

def test_transcript_shows_tools_and_results():
    text = critic.format_transcript([
        {"tool": "write_file", "result": "✅ Saved note.txt"},
        {"tool": "run_shell", "result": "ok"},
    ])
    assert "write_file" in text and "Saved note.txt" in text
    assert text.startswith("1.")


def test_transcript_truncates_long_results_and_old_steps():
    steps = [{"tool": f"t{i}", "result": "x" * 2000} for i in range(20)]
    text = critic.format_transcript(steps, limit=3)
    assert "8 earlier steps omitted" not in text  # 20 - 3 = 17
    assert "17 earlier steps omitted" in text
    assert "truncated" in text
    assert len(text) < 3000


def test_transcript_says_so_when_no_tools_ran():
    assert "no tools" in critic.format_transcript([])


def test_critic_prompt_carries_goal_transcript_and_answer():
    ai = ScriptedAI("VERDICT: ACCEPT")
    critic.review("Deploy the site", [{"tool": "run_shell", "result": "built"}], "Shipped it.", ai)

    prompt = ai.calls[0][0][0]["content"]
    assert "Deploy the site" in prompt
    assert "run_shell" in prompt and "built" in prompt
    assert "Shipped it." in prompt
    # The rule that stops the blocked-work loop must reach the critic.
    flat = " ".join(prompt.lower().split())
    assert "a blocked tool" in flat
    assert "missing credential or token" in flat
    assert "reporting a real blocker is a complete outcome" in flat


def test_critic_never_sees_the_agents_own_prose_as_evidence():
    """The transcript is tool results only — the agent can't vouch for itself."""
    ai = ScriptedAI("VERDICT: ACCEPT")
    critic.review("goal", [{"tool": "run_shell", "result": "real output"}], "answer", ai)
    prompt = ai.calls[0][0][0]["content"]
    assert "real output" in prompt
    assert "[TOOL_CALL]" not in prompt


# ── Run engine integration ──

def test_rejected_answer_goes_back_and_the_second_one_ships(monkeypatch):
    monkeypatch.setenv("CRITIC_MAX_ROUNDS", "2")
    ai = ScriptedAI(
        "I saved the report for you.",                      # unsupported claim
        "VERDICT: REVISE\nREASON: No write_file call in the transcript.",
        tool_call("write_file", filename="r.txt", content="data"),  # actually does it
        "Saved r.txt.",
        "VERDICT: ACCEPT",
    )
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("Write the report to a file")
    final = runner.execute_run(runner.get_run(run_id))

    assert final["status"] == "done"
    assert final["result"] == "Saved r.txt."
    assert "unresolved" not in final["result"]

    kinds = [e["kind"] for e in runner.get_events(run_id)]
    assert kinds == ["assistant", "critic", "assistant", "tool_result", "critic_ok", "final"]


def test_critic_calls_count_against_the_step_budget(monkeypatch):
    ai = ScriptedAI("Done.", "VERDICT: ACCEPT")
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("simple task")
    final = runner.execute_run(runner.get_run(run_id))
    # One agent step + one critic call, both real AI calls.
    assert final["steps_used"] == 2


def test_round_cap_ships_the_work_with_the_critique_attached(monkeypatch):
    monkeypatch.setenv("CRITIC_MAX_ROUNDS", "1")
    ai = ScriptedAI(
        "Done, I promise.",
        "VERDICT: REVISE\nREASON: Nothing in the transcript supports that.",
        "Still done.",
    )
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("do the thing")
    final = runner.execute_run(runner.get_run(run_id))

    assert final["status"] == "done"
    assert "Still done." in final["result"]
    # The concern survives into what the human reads, rather than vanishing.
    assert "unresolved review note" in final["result"]
    assert "Nothing in the transcript supports that." in final["result"]


def test_critic_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("CRITIC_ENABLED", "false")
    ai = ScriptedAI("Done.")
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("simple task")
    final = runner.execute_run(runner.get_run(run_id))

    assert final["steps_used"] == 1  # no extra critic call
    assert [e["kind"] for e in runner.get_events(run_id)] == ["final"]


def test_blocked_tool_report_is_not_sent_back_forever(monkeypatch):
    """
    An unattended run that correctly stopped at a blocked deploy is finished.
    The critic accepting that is the behaviour; this pins the round cap as the
    backstop if a critic ever disagrees anyway.
    """
    monkeypatch.setenv("CRITIC_MAX_ROUNDS", "2")
    monkeypatch.setenv("APPROVALS_ENABLED", "false")
    ai = ScriptedAI(
        tool_call("push_branch", branch="x"),
        "Could not deploy — the tool is blocked for unattended runs. Needs a human.",
        "VERDICT: REVISE\nREASON: The site was not deployed.",   # a bad critic
        "As I said: deploying is blocked here. A human has to run it.",
        "VERDICT: REVISE\nREASON: The site was still not deployed.",
        "Blocked. Nothing more I can do.",
    )
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("deploy the site", unattended=True)
    final = runner.execute_run(runner.get_run(run_id))

    assert final["status"] == "done"
    assert runner.critic_rounds_used(run_id) == 2  # capped, not endless
    assert "Blocked." in final["result"]


def test_resumed_run_replays_the_critique_it_was_sent_back_with(monkeypatch):
    monkeypatch.setenv("CRITIC_MAX_ROUNDS", "1")
    ai = ScriptedAI(
        "Done!",
        "VERDICT: REVISE\nREASON: Step two was never attempted.",
    )
    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")

    run_id = runner.enqueue_run("two-step job", max_steps=2)
    runner.execute_run(runner.get_run(run_id))

    messages = runner.rebuild_messages(runner.get_run(run_id))
    assert any("Step two was never attempted." in m["content"] for m in messages)
    # And the round already spent is remembered, so a resume can't reset the cap.
    assert runner.critic_rounds_used(run_id) == 1


# ── Interactive path ──

def test_interactive_critic_is_off_by_default():
    ai = ScriptedAI("Here you go.")
    result = agent.run_agent_loop(
        messages=[{"role": "user", "content": "hi"}],
        call_ai_fn=ai,
        system_prompt="Test bot.",
    )
    assert result == "Here you go."
    assert len(ai.calls) == 1  # no critic call


def test_interactive_critic_sends_a_reply_back_when_enabled(monkeypatch):
    monkeypatch.setenv("CRITIC_INTERACTIVE", "true")
    monkeypatch.setenv("CRITIC_MAX_ROUNDS", "2")
    ai = ScriptedAI(
        "I've noted that down.",
        "VERDICT: REVISE\nREASON: No remember call was made.",
        tool_call("remember", fact="likes tea", category="fact"),
        "Noted — you like tea.",
        "VERDICT: ACCEPT",
    )
    result = agent.run_agent_loop(
        messages=[{"role": "user", "content": "remember I like tea"}],
        call_ai_fn=ai,
        system_prompt="Test bot.",
        user_id="U1",
    )
    assert result == "Noted — you like tea."
    assert any("likes tea" in f for f in memory.get_facts("U1")["recent"])


def test_interactive_cap_attaches_the_unresolved_note(monkeypatch):
    monkeypatch.setenv("CRITIC_INTERACTIVE", "true")
    monkeypatch.setenv("CRITIC_MAX_ROUNDS", "1")
    ai = ScriptedAI(
        "All handled.",
        "VERDICT: REVISE\nREASON: You never checked the file.",
        "Still all handled.",
    )
    result = agent.run_agent_loop(
        messages=[{"role": "user", "content": "check the file"}],
        call_ai_fn=ai,
        system_prompt="Test bot.",
    )
    assert "Still all handled." in result
    assert "You never checked the file." in result
