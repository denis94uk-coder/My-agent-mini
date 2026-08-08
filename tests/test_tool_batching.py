"""
Several read-only tools from one response.

Two things met here. Every extra round trip re-sends the whole system prompt,
which is what free-tier token/minute limits actually bill for — three files
read one at a time cost three prompts. And the single-call parser was greedy,
so two [TOOL_CALL] blocks in one response matched from the first opening tag
to the last closing tag, parsed as nothing, and were treated as a final
answer: the model's narration was posted while neither tool ran.

Only READ-tier tools batch. A read is idempotent and side-effect-free, so
ordering and partial failure are harmless; a write needs its result seen
before the next choice, and an EXTERNAL tool needs its own approval decision.
"""

import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent
import governor
import memory
import tools


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "memory.db")
    monkeypatch.setenv("OWNER_SLACK_ID", "default")
    yield


def call(name: str, **args) -> str:
    return f'[TOOL_CALL]\n{{"tool": "{name}", "args": {json.dumps(args)}}}\n[/TOOL_CALL]'


def reply_with(*responses):
    queue = list(responses)
    def call_ai(messages, prompt):
        return queue.pop(0) if queue else "Finished."
    return call_ai


def fake_tools(monkeypatch, log):
    def run_tool(name, args):
        log.append((name, dict(args)))
        return f"result of {name}"
    monkeypatch.setattr(tools, "run_tool", run_tool)


# ── Parsing ──

def test_two_blocks_both_parse():
    """The regression: greedy matching returned None and neither tool ran."""
    text = call("read_file", filename="a.txt") + "\n" + call("read_file", filename="b.txt")
    calls = agent.parse_tool_calls(text)
    assert [c["args"]["filename"] for c in calls] == ["a.txt", "b.txt"]


def test_braces_inside_arguments_do_not_end_a_block():
    text = call("run_python", code="d = {'a': {'b': 1}}")
    assert agent.parse_tool_calls(text)[0]["args"]["code"] == "d = {'a': {'b': 1}}"


def test_a_missing_closing_tag_still_parses():
    text = '[TOOL_CALL]\n{"tool": "list_files", "args": {}}'
    assert agent.parse_tool_calls(text)[0]["tool"] == "list_files"


def test_a_missing_closing_tag_between_two_blocks():
    text = '[TOOL_CALL]{"tool": "list_files", "args": {}}\n' + call("read_file", filename="a")
    assert [c["tool"] for c in agent.parse_tool_calls(text)] == ["list_files", "read_file"]


def test_prose_around_blocks_is_ignored():
    text = "First I'll check both.\n" + call("list_files") + "\nand also\n" + call("server_health")
    assert [c["tool"] for c in agent.parse_tool_calls(text)] == ["list_files", "server_health"]


def test_no_blocks_gives_an_empty_list():
    assert agent.parse_tool_calls("just an answer") == []


def test_malformed_json_is_skipped_not_fatal():
    text = "[TOOL_CALL]{not json}[/TOOL_CALL]" + call("list_files")
    assert [c["tool"] for c in agent.parse_tool_calls(text)] == ["list_files"]


def test_parse_tool_call_still_returns_the_first():
    """The single-call helper stays the API the run engine already uses."""
    text = call("read_file", filename="a") + call("read_file", filename="b")
    assert agent.parse_tool_call(text)["args"]["filename"] == "a"
    assert agent.parse_tool_call("nothing here") is None


# ── Batching ──

def test_several_reads_run_in_one_step(monkeypatch):
    log = []
    fake_tools(monkeypatch, log)
    messages = [{"role": "user", "content": "check both"}]
    response = call("list_files") + call("server_health")

    outcome = agent.execute_step(messages, reply_with(response), "prompt")

    assert outcome.kind == "tool"
    assert [name for name, _ in log] == ["list_files", "server_health"]
    assert [r["tool"] for r in outcome.batch] == ["list_files", "server_health"]


def test_the_batch_feeds_back_one_turn_carrying_every_result(monkeypatch):
    """One turn, not one per tool — that is where the saved call comes from."""
    fake_tools(monkeypatch, [])
    messages = [{"role": "user", "content": "check both"}]

    agent.execute_step(
        messages, reply_with(call("list_files") + call("server_health")), "prompt"
    )

    assert len(messages) == 3  # original + assistant + one combined result
    combined = messages[-1]["content"]
    assert "result of list_files" in combined
    assert "result of server_health" in combined


def test_writes_never_batch(monkeypatch):
    """
    A write must be seen before the next choice is made, so only the first
    runs and the model decides again with that result in hand.
    """
    log = []
    fake_tools(monkeypatch, log)
    messages = [{"role": "user", "content": "save both"}]
    response = call("write_file", filename="a", content="x") + call("write_file", filename="b", content="y")

    agent.execute_step(messages, reply_with(response), "prompt")

    assert [name for name, _ in log] == ["write_file"]


def test_a_read_mixed_with_a_write_does_not_batch(monkeypatch):
    log = []
    fake_tools(monkeypatch, log)
    messages = [{"role": "user", "content": "read then save"}]

    agent.execute_step(
        messages,
        reply_with(call("list_files") + call("write_file", filename="a", content="x")),
        "prompt",
    )

    assert [name for name, _ in log] == ["list_files"]


def test_external_tools_never_batch(monkeypatch):
    """An EXTERNAL tool needs its own approval decision, which a batch cannot express."""
    log = []
    fake_tools(monkeypatch, log)
    messages = [{"role": "user", "content": "ship it"}]

    agent.execute_step(
        messages,
        reply_with(call("push_branch", repo="r", branch_name="b") + call("list_files")),
        "prompt",
    )

    assert [name for name, _ in log] == ["push_branch"]


def test_a_single_read_is_not_treated_as_a_batch(monkeypatch):
    fake_tools(monkeypatch, [])
    messages = [{"role": "user", "content": "list"}]

    outcome = agent.execute_step(messages, reply_with(call("list_files")), "prompt")

    assert outcome.batch == []
    assert outcome.tool_name == "list_files"


def test_every_batchable_tool_named_in_the_prompt_is_read_tier():
    """
    The prompt lists which tools may be batched. If that list drifts from the
    tier table, the model is invited to batch a write and the batch silently
    refuses to form.
    """
    prompt = agent.get_agent_system_prompt("base")
    listed = prompt.split("ONE EXCEPTION")[1].split("Only batch reads")[0]
    named = [t for t in governor.TOOL_TIERS if f"{t}," in listed or f"{t})" in listed]
    assert len(named) >= 10, "prompt should name the batchable read tools"
    for name in named:
        assert governor.tier_of(name) == governor.READ, f"{name} is named but not READ"


# ── Durability ──

def test_the_batch_persists_as_one_event_so_resume_rebuilds_it(monkeypatch):
    """
    runner replays a step as tool_result_message(name, content). A batch must
    go through that same shape, or a resumed run rebuilds a transcript the
    live path never produced.
    """
    fake_tools(monkeypatch, [])
    messages = [{"role": "user", "content": "check both"}]

    outcome = agent.execute_step(
        messages, reply_with(call("list_files") + call("server_health")), "prompt"
    )

    replayed = agent.tool_result_message(outcome.tool_name, outcome.tool_result)
    assert replayed == messages[-1]["content"]


# ── The wrap-up call at max iterations ──

def test_a_usable_last_response_is_sent_instead_of_buying_a_wrap_up(monkeypatch):
    """
    Hitting the iteration cap used to always cost one more full call to say
    "give your final answer now". When the last response already carries the
    answer beside its tool block, that call buys a rephrasing.
    """
    fake_tools(monkeypatch, [])
    prose = (
        "Here is what I found across the files: the configuration is loaded at "
        "startup, the worker count is two, and the transcript budget is applied "
        "before every call rather than once per run. That accounts for the "
        "behaviour you asked about, and nothing further needs checking here."
    )
    calls = []

    def call_ai(messages, prompt):
        calls.append(1)
        return prose + call("list_files")

    result = agent.run_agent_loop(
        messages=[{"role": "user", "content": "investigate"}],
        call_ai_fn=call_ai,
        system_prompt="Test bot.",
        user_id="U1",
    )

    assert result == prose.strip()
    assert len(calls) == agent.MAX_ITERATIONS, "no extra wrap-up call"


def test_a_bare_tool_call_still_buys_the_wrap_up(monkeypatch):
    """With no prose to salvage there is nothing to send, so the call is earned."""
    fake_tools(monkeypatch, [])
    calls = []

    def call_ai(messages, prompt):
        calls.append(1)
        if len(calls) > agent.MAX_ITERATIONS:
            return "Final summary."
        return call("list_files")

    result = agent.run_agent_loop(
        messages=[{"role": "user", "content": "investigate"}],
        call_ai_fn=call_ai,
        system_prompt="Test bot.",
        user_id="U1",
    )

    assert result == "Final summary."
    assert len(calls) == agent.MAX_ITERATIONS + 1
