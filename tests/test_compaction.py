"""
Tests for context compaction — the fold that keeps a long run inside the
model's context window, and keeps a *resumed* run from paying for it twice.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent
import memory
import runner


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "memory.db")
    # The owner lock fails closed; these tests exercise execution, not auth.
    monkeypatch.setenv("OWNER_SLACK_ID", "default")
    monkeypatch.setattr(runner, "_CALL_AI", None)
    monkeypatch.setattr(runner, "_POST_MESSAGE", None)
    monkeypatch.setenv("CRITIC_ENABLED", "false")
    yield


def tool_call(name: str, **args) -> str:
    import json
    return f'[TOOL_CALL]\n{{"tool": "{name}", "args": {json.dumps(args)}}}\n[/TOOL_CALL]'


def transcript(pairs: int) -> list[dict]:
    """A goal plus `pairs` assistant/tool-result exchanges."""
    messages = [{"role": "user", "content": "the original goal"}]
    for i in range(pairs):
        messages.append({"role": "assistant", "content": f"calling tool {i}"})
        messages.append({
            "role": "user",
            "content": agent.tool_result_message("run_shell", f"output {i} " + "x" * 500),
        })
    return messages


# ── The fold itself ──

def test_short_transcripts_are_left_alone():
    assert runner.compact_messages(transcript(1)) is None


def test_fold_keeps_the_goal_and_the_recent_tail():
    messages = transcript(10)
    folded, note = runner.compact_messages(messages)

    assert folded[0] == messages[0]                       # goal survives verbatim
    assert folded[1]["content"] == note                   # the progress note
    assert folded[2:] == messages[-runner.COMPACTION_KEEP_RECENT:]
    assert len(folded) == runner.COMPACTION_KEEP_RECENT + 2


def test_fold_actually_shrinks_the_context():
    messages = transcript(20)
    folded, _ = runner.compact_messages(messages)
    assert runner._messages_size(folded) < runner._messages_size(messages) / 2


def test_fold_uses_the_ai_summary_when_one_is_available():
    calls = []

    def summarizer(messages, prompt):
        calls.append(messages[0]["content"])
        return "Ran 8 shell commands; build passes; deploy still pending."

    _, note = runner.compact_messages(transcript(10), summarizer)
    assert "build passes" in note
    assert len(calls) == 1


def test_fold_falls_back_to_a_deterministic_digest_when_the_ai_fails():
    """The router being down is exactly when a long run needs to keep going."""
    def broken(messages, prompt):
        raise RuntimeError("all providers cooling down")

    _, note = runner.compact_messages(transcript(10), broken)
    assert "run_shell" in note          # built from the transcript, no AI needed
    assert "output 0" in note


def test_fold_falls_back_on_a_router_error_string():
    _, note = runner.compact_messages(transcript(10), lambda m, p: "❌ All routes failed")
    assert "run_shell" in note


def test_fold_falls_back_on_an_empty_summary():
    _, note = runner.compact_messages(transcript(10), lambda m, p: "   ")
    assert "run_shell" in note


def test_note_tells_the_model_not_to_redo_the_work():
    _, note = runner.compact_messages(transcript(10))
    assert "do not repeat" in note.lower()


# ── Inside a run ──

def test_long_run_folds_its_context_and_records_it(monkeypatch):
    monkeypatch.setenv("RUN_CONTEXT_LIMIT_CHARS", "2500")

    replies = [tool_call("run_python", code="print('x' * 400)") for _ in range(6)]
    replies.append("All done.")
    queue = list(replies)

    def ai(messages, prompt):
        if "progress" in messages[0]["content"].lower() and len(messages) == 1:
            return "Summarised progress."          # the compaction call
        return queue.pop(0) if queue else "All done."

    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")
    run_id = runner.enqueue_run("long job", max_steps=12)
    final = runner.execute_run(runner.get_run(run_id))

    assert final["status"] == "done"
    kinds = [e["kind"] for e in runner.get_events(run_id)]
    assert "compaction" in kinds


def test_resume_rebuilds_from_the_fold_not_the_full_history():
    """
    The saving has to survive a restart. If a resumed run replayed everything
    before the fold, the compaction would be paid again on every recovery and
    the context would be just as large as before.
    """
    run_id = runner.enqueue_run("big goal")
    runner.add_event(run_id, "assistant", "step one")
    runner.add_event(run_id, "tool_result", "x" * 3000, name="run_shell")
    runner.add_event(run_id, "compaction", "[PROGRESS SO FAR] did step one")
    runner.add_event(run_id, "assistant", "step two")
    runner.add_event(run_id, "tool_result", "step two output", name="run_shell")

    messages = runner.rebuild_messages(runner.get_run(run_id))

    assert messages[0]["content"] == "big goal"
    assert messages[1]["content"] == "[PROGRESS SO FAR] did step one"
    assert not any("x" * 3000 in m["content"] for m in messages)   # pre-fold history gone
    assert any("step two output" in m["content"] for m in messages)  # post-fold kept
    assert len(messages) == 4


def test_compaction_counts_against_the_step_budget(monkeypatch):
    monkeypatch.setenv("RUN_CONTEXT_LIMIT_CHARS", "3000")
    ai_calls = []

    def ai(messages, prompt):
        ai_calls.append(1)
        return "Done."

    runner.configure(call_ai_fn=ai, system_prompt="Test bot.")
    run_id = runner.enqueue_run("x" * 600)
    runner.add_event(run_id, "assistant", "a" * 300)
    runner.add_event(run_id, "tool_result", "b" * 300, name="run_shell")
    runner.add_event(run_id, "assistant", "c" * 300)
    runner.add_event(run_id, "tool_result", "d" * 300, name="run_shell")
    runner.add_event(run_id, "assistant", "e" * 300)
    runner.add_event(run_id, "tool_result", "f" * 300, name="run_shell")
    runner.add_event(run_id, "assistant", "g" * 300)
    runner.add_event(run_id, "tool_result", "h" * 300, name="run_shell")

    final = runner.execute_run(runner.get_run(run_id))
    # One summariser call + one agent step, both real AI calls.
    assert final["steps_used"] == 2
    assert len(ai_calls) == 2


# ── Context budget ──

def test_system_prompt_stays_within_a_sane_size():
    """
    The system prompt is sent on every call and silently grows every time a
    tool or playbook is added — with the operating manual attached it is
    ~26k chars (~6.4k tokens), still larger than the whole transcript budget.
    Past ~45k the default config stops fitting a 16k-context route, so this
    fails before a user discovers it as a mid-run provider rejection.

    It got there once: a free-tier route rejected the assembled payload with
    HTTP 413, and since it was the only route configured, the bot had none.
    """
    prompt = agent.get_agent_system_prompt("You are a test bot.")
    assert len(prompt) < 45_000, (
        f"system prompt has grown to {len(prompt):,} chars "
        f"(~{len(prompt) // 4:,} tokens) — re-check the context budget"
    )


def test_worst_case_context_is_documented_accurately():
    """Pins the arithmetic the .env.example and runner docstring state."""
    prompt = agent.get_agent_system_prompt("You are a test bot.")
    worst_case_tokens = (len(prompt) + runner.context_limit_chars()) // 4
    # The docs claim >= 16k context is required at the default budget.
    assert 8_000 < worst_case_tokens < 16_000
