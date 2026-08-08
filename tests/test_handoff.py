"""
Handing an over-long interactive turn to the run engine.

MAX_ITERATIONS is a limit on one chat reply, not evidence a task is
impossible. A task needing eleven steps used to end at ten with a partial
answer and no way to continue — the user's only recourse was to re-ask and pay
for the first ten steps over again. The run engine is built for exactly this
work: durable, resumable, budgeted, and not holding up the chat.

The rules worth pinning are the ones that keep this from becoming a way to
spawn runs by accident: the partial answer is never replaced by the handoff
note, a refused handoff is silent, and a background run can never reach this
code to spawn another.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent
import tools


def tool_call(name, **args):
    import json
    return f'[TOOL_CALL]{json.dumps({"tool": name, "args": args})}[/TOOL_CALL]'


@pytest.fixture
def spawned(monkeypatch):
    """Capture handoffs only — the loop's own tool calls go through the same
    entry point and would otherwise drown them."""
    handoffs = []

    def run_tool(name, args):
        if name == "start_background_run":
            handoffs.append(dict(args))
            return "🚀 Started background run #7"
        return f"result of {name}"

    monkeypatch.setattr(tools, "run_tool", run_tool)
    return handoffs


def _loop(call_ai, **kwargs):
    return agent.run_agent_loop(
        messages=[{"role": "user", "content": "do the long thing"}],
        call_ai_fn=call_ai,
        system_prompt="Test bot.",
        user_id=kwargs.pop("user_id", "U_OWNER"),
        **kwargs,
    )


def _never_finishes(messages, prompt):
    if "give your final answer now" in messages[-1]["content"]:
        return "Partial answer."
    return tool_call("list_files")


# ── The handoff ──

def test_hitting_the_cap_starts_a_background_run(spawned):
    result = _loop(_never_finishes)

    assert len(spawned) == 1
    assert "background run" in result


def test_the_partial_answer_is_kept_not_replaced(spawned):
    """INVARIANT: the work up to the cap was paid for and is often most of the
    answer. The handoff is an addition to the reply, never a substitute."""
    result = _loop(_never_finishes)

    assert result.startswith("Partial answer.")


def test_the_run_goal_carries_the_original_request(spawned):
    _loop(_never_finishes)

    goal = spawned[0]["goal"]
    assert "do the long thing" in goal


def test_the_run_goal_carries_what_was_already_done(spawned):
    """Without the transcript the continuation restarts from scratch and pays
    for the same ten steps again — the exact problem this fixes."""
    _loop(_never_finishes)

    goal = spawned[0]["goal"]
    assert "list_files" in goal
    assert "do not repeat" in goal.lower()


def test_the_run_goal_carries_the_partial_answer(spawned):
    """So the continuation reports only what is new, rather than repeating to
    the user what they just read."""
    _loop(_never_finishes)

    assert "Partial answer." in spawned[0]["goal"]


def test_no_ai_call_is_bought_to_build_the_digest(spawned):
    """The digest is assembled from the transcript. Summarising it with the
    model would mean paying for an extra call at exactly the moment the turn
    has already proved expensive."""
    calls = []

    def counting(messages, prompt):
        calls.append(1)
        if "give your final answer now" in messages[-1]["content"]:
            return "Partial answer."
        return tool_call("list_files")

    _loop(counting)

    assert len(calls) == agent.MAX_ITERATIONS + 1  # the wrap-up call, and no more


def test_the_conversation_is_passed_through_so_the_result_lands_in_thread(spawned):
    _loop(_never_finishes, conv_key="C123:456")

    assert spawned[0]["_conv_key"] == "C123:456"


def test_the_requesting_user_is_passed_so_the_owner_lock_applies(spawned):
    """Without this, run_tool sees no requesting user and refuses — or worse,
    a future change makes it fail open for a caller with no user at all."""
    _loop(_never_finishes, user_id="U_SOMEONE")

    assert spawned[0]["_requesting_user_id"] == "U_SOMEONE"


# ── When it must not happen ──

def test_a_refused_handoff_is_silent(monkeypatch):
    """INVARIANT: a non-owner cannot commit worker threads and metered calls to
    autonomous work. Their partial answer stands on its own — telling them the
    bot would have continued but may not is noise they cannot act on."""
    monkeypatch.setattr(tools, "run_tool", lambda name, args: "❌ Not authorized: owner only.")

    result = _loop(_never_finishes)

    assert result == "Partial answer."
    assert "background" not in result.lower()


def test_a_raising_handoff_does_not_lose_the_answer(monkeypatch):
    """Failing to hand off is a worse answer, not a failed one."""
    def explode(name, args):
        if name == "start_background_run":
            raise RuntimeError("the queue is on fire")
        return f"result of {name}"

    monkeypatch.setattr(tools, "run_tool", explode)

    assert _loop(_never_finishes) == "Partial answer."


def test_the_handoff_can_be_switched_off(monkeypatch, spawned):
    monkeypatch.setenv("HANDOFF_ON_MAX_ITERATIONS", "0")

    result = _loop(_never_finishes)

    assert spawned == []
    assert result == "Partial answer."


def test_a_normal_answer_never_hands_off(spawned):
    """The cap is the trigger. A loop that finishes on its own must not queue
    background work nobody asked for."""
    result = _loop(lambda messages, prompt: "Done, here is the answer.")

    assert spawned == []
    assert result == "Done, here is the answer."


def test_an_empty_goal_does_not_hand_off(spawned):
    """Nothing to continue, and a run with an empty goal burns a worker to
    discover that."""
    result = agent.run_agent_loop(
        messages=[{"role": "user", "content": "   "}],
        call_ai_fn=_never_finishes,
        system_prompt="Test bot.",
        user_id="U_OWNER",
    )

    assert spawned == []
    assert result == "Partial answer."


def test_the_run_engine_cannot_reach_this_code():
    """INVARIANT: no recursive spawning. The run engine drives execute_step
    directly and never calls run_agent_loop, which is the same reason
    start_background_run is in UNATTENDED_BLOCKED_TOOLS. If a future change
    routes runs through run_agent_loop, this fails and the block list has to
    be revisited first."""
    import runner

    source = Path(runner.__file__).read_text()
    assert "run_agent_loop" not in source.split('"""', 2)[-1], (
        "runner now calls run_agent_loop — a background run could hand off to "
        "another background run"
    )
    assert "start_background_run" in tools.UNATTENDED_BLOCKED_TOOLS
