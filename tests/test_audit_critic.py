"""
Phase 2.7 — the critic gate.

Invariants under test:
  • Fails OPEN in every direction: raises, garbage, timeout, provider error —
    all ACCEPT, all logged. A broken grader must not become a broken agent.
  • The critic judges the tool transcript, never the agent's own prose.
  • The round cap ships the work WITH the unresolved critique attached.
  • Critic calls are charged to the run's step budget.
  • Off by default for live chat, on for unattended runs.

`xfail(strict=True)` marks a CONFIRMED DEFECT — see it fail with
`pytest --runxfail`.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_support import ScriptedAI, audit_env, run_id_of, tool_call  # noqa: F401

import critic
import runner


# ── fail open ──

def test_fails_open_when_the_critic_raises(audit_env, caplog):
    def boom(messages, prompt=None):
        raise RuntimeError("provider exploded")
    assert critic.review("goal", [{"tool": "x", "result": "y"}], "final", boom).accepted
    assert "accepting" in caplog.text.lower(), "the fail-open was not logged"


def test_fails_open_on_a_timeout(audit_env):
    def timeout(messages, prompt=None):
        raise TimeoutError("read timed out")
    assert critic.review("g", [], "f", timeout).accepted


@pytest.mark.parametrize("reply", [
    "", "   ", "¯\\_(ツ)_/¯", "VERDICT: MAYBE", "REVISE", "VERDICT: REVISE",
    "VERDICT: REVISE\nREASON:", "<html>502 Bad Gateway</html>", None,
])
def test_fails_open_on_unparseable_output(audit_env, reply):
    assert critic.review("g", [], "final", lambda m, p=None: reply).accepted, reply


def test_fails_open_on_a_router_error_string(audit_env):
    """'❌ All AI providers failed' is a router failure, not a verdict."""
    assert critic.review("g", [], "f",
                         lambda m, p=None: "❌ All AI providers failed").accepted


def test_an_empty_final_answer_is_accepted_without_an_ai_call(audit_env):
    calls = []
    critic.review("g", [], "", lambda m, p=None: calls.append(1) or "VERDICT: REVISE")
    assert calls == [], "the critic spent a call on an empty answer"


def test_a_well_formed_revise_is_still_honoured(audit_env):
    """Fail-open must not degrade into never revising."""
    verdict = critic.review("g", [], "final",
                            lambda m, p=None: "VERDICT: REVISE\nREASON: you never saved it.")
    assert verdict.kind == critic.REVISE and "never saved" in verdict.reason


# ── evidence discipline ──

def test_the_critic_never_sees_the_agents_prose_as_evidence(audit_env):
    """A confident false claim with an empty transcript: the answer under review
    appears once, and the evidence section says plainly that no tools ran."""
    seen = {}

    def spy(messages, prompt=None):
        seen["prompt"] = messages[0]["content"]
        return "VERDICT: REVISE\nREASON: the transcript shows no write."

    verdict = critic.review(
        "write report.md",
        [],
        "I have written report.md, verified the contents, and double-checked it.",
        spy)
    prompt = seen["prompt"]
    assert "(no tools were used" in prompt
    assert prompt.count("I have written report.md") == 1, (
        "the agent's claim was fed in twice — once as evidence")
    assert verdict.kind == critic.REVISE


def test_tool_results_are_the_evidence_and_are_truncated_not_dropped(audit_env):
    steps = [{"tool": f"t{i}", "result": f"r{i}"} for i in range(20)]
    rendered = critic.format_transcript(steps, limit=12)
    assert "8 earlier steps omitted" in rendered
    assert "t19" in rendered and "t0" not in rendered.split("\n")[0]


def test_a_failed_tool_is_visible_to_the_critic(audit_env):
    rendered = critic.format_transcript([{"tool": "push_branch",
                                          "result": "❌ Push failed: no credentials"}])
    assert "Push failed" in rendered


# ── round cap and budget ──

def test_round_cap_ships_the_work_with_the_critique_attached(audit_env, monkeypatch):
    monkeypatch.setenv("CRITIC_ENABLED", "true")
    monkeypatch.setenv("CRITIC_MAX_ROUNDS", "1")
    monkeypatch.setattr(runner, "_CALL_AI", ScriptedAI(
        "first answer",
        "VERDICT: REVISE\nREASON: you never actually saved the file.",
        "second answer",
    ))
    row = runner.enqueue_run("save the file", owner_user_id="U_OWNER", unattended=True)
    final = runner.execute_run(runner.get_run(run_id_of(row)))
    assert final["status"] == "done"
    assert "second answer" in final["result"], "the work was eaten by the gate"
    assert "unresolved review note" in final["result"]
    assert "you never actually saved the file" in final["result"]


def test_a_stuck_critic_cannot_loop_forever(audit_env, monkeypatch):
    """Every round is a REVISE; the cap must still land and ship."""
    monkeypatch.setenv("CRITIC_ENABLED", "true")
    monkeypatch.setenv("CRITIC_MAX_ROUNDS", "2")

    def never_happy(messages, prompt=None):
        text = str(messages[-1].get("content", ""))
        if "CRITIC" in text or "answer" not in text:
            return "an answer"
        return "VERDICT: REVISE\nREASON: still not done."

    monkeypatch.setattr(runner, "_CALL_AI", never_happy)
    row = runner.enqueue_run("do it", owner_user_id="U_OWNER", unattended=True,
                             max_steps=12)
    final = runner.execute_run(runner.get_run(run_id_of(row)))
    assert final["status"] == "done"
    assert runner.critic_rounds_used(run_id_of(row)) <= 2


def test_critic_calls_are_charged_to_the_step_budget(audit_env, monkeypatch):
    monkeypatch.setenv("CRITIC_ENABLED", "true")
    monkeypatch.setattr(runner, "_CALL_AI", ScriptedAI("answer", "VERDICT: ACCEPT"))
    row = runner.enqueue_run("do it", owner_user_id="U_OWNER", unattended=True)
    final = runner.execute_run(runner.get_run(run_id_of(row)))
    assert final["steps_used"] >= 2, "the critic's AI call was invisible to the budget"


def test_the_accept_path_is_recorded_in_the_durable_transcript(audit_env, monkeypatch):
    monkeypatch.setenv("CRITIC_ENABLED", "true")
    monkeypatch.setattr(runner, "_CALL_AI", ScriptedAI("answer", "VERDICT: ACCEPT"))
    row = runner.enqueue_run("do it", owner_user_id="U_OWNER", unattended=True)
    runner.execute_run(runner.get_run(run_id_of(row)))
    kinds = [e["kind"] for e in runner.get_events(run_id_of(row))]
    assert "critic_ok" in kinds


def test_a_critic_round_replays_on_resume(audit_env, monkeypatch):
    """Otherwise a resumed run re-delivers the answer the critic rejected."""
    monkeypatch.setenv("CRITIC_ENABLED", "true")
    monkeypatch.setenv("CRITIC_MAX_ROUNDS", "1")
    monkeypatch.setattr(runner, "_CALL_AI", ScriptedAI(
        "first answer", "VERDICT: REVISE\nREASON: not saved.", "second answer"))
    row = runner.enqueue_run("save it", owner_user_id="U_OWNER", unattended=True)
    run_id = run_id_of(row)
    runner.execute_run(runner.get_run(run_id))
    replayed = "\n".join(str(m) for m in runner.rebuild_messages(runner.get_run(run_id)))
    assert "not saved" in replayed, "the critic round vanished from the replay"


# ── configuration defaults ──

def test_defaults_on_for_runs_off_for_live_chat(audit_env, monkeypatch):
    monkeypatch.delenv("CRITIC_ENABLED", raising=False)
    monkeypatch.delenv("CRITIC_INTERACTIVE", raising=False)
    assert critic.enabled() is True
    assert critic.interactive_enabled() is False
    assert critic.max_rounds() == 2


def test_max_rounds_zero_disables_the_gate_without_eating_work(audit_env, monkeypatch):
    monkeypatch.setenv("CRITIC_ENABLED", "true")
    monkeypatch.setenv("CRITIC_MAX_ROUNDS", "0")
    monkeypatch.setattr(runner, "_CALL_AI", ScriptedAI("the answer"))
    row = runner.enqueue_run("do it", owner_user_id="U_OWNER", unattended=True)
    final = runner.execute_run(runner.get_run(run_id_of(row)))
    assert final["result"].startswith("the answer")


def test_a_reported_blocker_is_accepted_without_relying_on_the_prompt(audit_env):
    """FIXED (was FINDING C1): the refusal markers in the transcript are checked
    in code, so a weak critic route can no longer demand impossible work."""
    steps = [{"tool": "push_branch",
              "result": "❌ Not authorized: 'push_branch' can only be used by the owner."}]
    stubborn = lambda m, p=None: "VERDICT: REVISE\nREASON: you did not push the branch."  # noqa: E731
    verdict = critic.review("push the branch", steps,
                            "I could not push: the tool is owner-only. Reporting instead.",
                            stubborn)
    assert verdict.accepted, "a correctly-reported blocker was sent back for revision"
