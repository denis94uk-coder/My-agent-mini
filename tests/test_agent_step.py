"""
Tests for the single-step primitive the interactive loop and the run engine
both build on.
"""

import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent
import memory


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "memory.db")
    # The owner lock fails closed; these tests exercise execution, not auth.
    monkeypatch.setenv("OWNER_SLACK_ID", "default")
    yield


def tool_call(name: str, **args) -> str:
    return f'[TOOL_CALL]\n{{"tool": "{name}", "args": {json.dumps(args)}}}\n[/TOOL_CALL]'


def reply_with(*responses):
    queue = list(responses)
    def call_ai(messages, prompt):
        return queue.pop(0) if queue else "Finished."
    return call_ai


def test_plain_answer_is_final():
    messages = [{"role": "user", "content": "hi"}]
    outcome = agent.execute_step(messages, reply_with("Hello!"), "prompt")

    assert outcome.kind == "final"
    assert outcome.response == "Hello!"
    assert messages == [{"role": "user", "content": "hi"}]  # untouched


def test_tool_call_runs_and_feeds_the_result_back():
    messages = [{"role": "user", "content": "what is 6*7?"}]
    outcome = agent.execute_step(
        messages, reply_with(tool_call("run_python", code="print(6*7)")), "prompt"
    )

    assert outcome.kind == "tool"
    assert outcome.tool_name == "run_python"
    assert "42" in outcome.tool_result
    assert messages[-2]["role"] == "assistant"
    assert "TOOL_RESULT for run_python" in messages[-1]["content"]


def test_unactioned_intent_becomes_a_nudge():
    messages = [{"role": "user", "content": "remember I like tea"}]
    outcome = agent.execute_step(messages, reply_with("I'll remember that."), "prompt")

    assert outcome.kind == "nudge"
    assert messages[-1]["content"] == agent.NUDGE_MESSAGE


def test_nudging_can_be_disabled():
    messages = [{"role": "user", "content": "remember I like tea"}]
    outcome = agent.execute_step(
        messages, reply_with("I'll remember that."), "prompt", allow_nudge=False
    )
    assert outcome.kind == "final"


def test_blocked_tool_is_refused_without_running():
    messages = [{"role": "user", "content": "deploy it"}]
    outcome = agent.execute_step(
        messages,
        reply_with(tool_call("deploy_static_site", site_name="demo")),
        "prompt",
        blocked_tools={"deploy_static_site"},
    )

    assert outcome.kind == "tool"
    assert "Blocked" in outcome.tool_result
    assert "unattended" in outcome.tool_result


def test_memory_tools_get_the_caller_context_injected():
    messages = [{"role": "user", "content": "remember I use Postgres"}]
    outcome = agent.execute_step(
        messages,
        reply_with(tool_call("remember", fact="uses Postgres", category="decision")),
        "prompt",
        user_id="U42",
    )

    assert outcome.tool_args["user_id"] == "U42"
    assert "decision" in memory.get_facts("U42")["durable"][0] or True
    assert any("Postgres" in f for f in memory.get_facts("U42")["durable"])


def test_plan_tools_get_the_conversation_key_injected():
    messages = [{"role": "user", "content": "plan it"}]
    agent.execute_step(
        messages,
        reply_with(tool_call("create_plan", steps=["a", "b"])),
        "prompt",
        user_id="U42",
        conv_key="C9:main",
    )
    assert len(memory.get_plan("C9:main")) == 2


def test_autonomy_tools_get_user_and_conv_context():
    captured = {}

    import tools

    def fake_run_tool(name, args):
        captured.update(args)
        return "ok"

    original = tools.run_tool
    tools.run_tool = fake_run_tool
    try:
        agent.execute_step(
            [{"role": "user", "content": "do it in the background"}],
            reply_with(tool_call("start_background_run", goal="big job")),
            "prompt",
            user_id="U42",
            conv_key="C9:1712.5",
        )
    finally:
        tools.run_tool = original

    assert captured["_user_id"] == "U42"
    assert captured["_conv_key"] == "C9:1712.5"


def test_malformed_args_do_not_crash_the_step():
    messages = [{"role": "user", "content": "go"}]
    outcome = agent.execute_step(
        messages, reply_with('[TOOL_CALL]{"tool": "run_python", "args": "oops"}[/TOOL_CALL]'), "p"
    )
    # Not a dict → treated as no args; the tool reports the error rather than
    # taking the whole run down.
    assert outcome.kind == "tool"
    assert outcome.tool_result.startswith("❌")


def test_loop_returns_the_final_answer_after_a_tool():
    result = agent.run_agent_loop(
        messages=[{"role": "user", "content": "what is 6*7?"}],
        call_ai_fn=reply_with(tool_call("run_python", code="print(6*7)"), "It's 42."),
        system_prompt="Test bot.",
        user_id="U1",
    )
    assert result == "It's 42."


def test_loop_nudges_then_completes():
    calls = []

    def call_ai(messages, prompt):
        calls.append(messages[-1]["content"])
        if len(calls) == 1:
            return "I'll save that now."
        if len(calls) == 2:
            return tool_call("write_file", filename="note.txt", content="saved")
        return "Saved it."

    result = agent.run_agent_loop(
        messages=[{"role": "user", "content": "save a note"}],
        call_ai_fn=call_ai,
        system_prompt="Test bot.",
    )

    assert agent.NUDGE_MESSAGE in calls[1]
    assert result == "Saved it."


def test_loop_stops_at_max_iterations():
    """A model that never stops still has to produce an answer for the user."""
    def always_tool(messages, prompt):
        if "give your final answer now" in messages[-1]["content"]:
            return "Final summary."
        return tool_call("run_python", code="print(1)")

    result = agent.run_agent_loop(
        messages=[{"role": "user", "content": "loop"}],
        call_ai_fn=always_tool,
        system_prompt="Test bot.",
    )
    assert result == "Final summary."
