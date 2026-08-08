"""
Task-class routing — sending the critic and the summariser somewhere better
than the agent loop.

The three call classes are not equally demanding. The critic re-reads a whole
transcript and rules on whether work is finished; the summariser writes a fold
that is then replayed on every later step and again on resume; the agent loop's
tool-argument steps are the most forgiving of a weak model. Routing all three
identically is what the open roadmap item "critic/summariser on a stronger
route than the agent itself" was about.

The invariant that matters most here is that this is a *preference*, not a
restriction. It reorders the route list and never filters it, so a preferred
route that is cooling down, rate-limited or past its daily cap loses its place
at the front rather than taking the call down with it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ai_routing
import governor


@pytest.fixture(autouse=True)
def _clean_routing_env(monkeypatch):
    for task in governor.TASK_CLASSES:
        monkeypatch.delenv(f"ROUTER_ROUTE_{task.upper()}", raising=False)
    yield


# ── Preference parsing ──

def test_an_unconfigured_class_expresses_no_preference():
    """INVARIANT: with nothing configured, routing is exactly what it was."""
    for task in governor.TASK_CLASSES:
        assert governor.task_route_preference(task) == []
        assert governor.task_route_rank("Groq", task) == governor._UNRANKED


def test_preference_is_read_per_class(monkeypatch):
    monkeypatch.setenv("ROUTER_ROUTE_CRITIC", "groq, gemini")

    assert governor.task_route_preference(governor.TASK_CRITIC) == ["groq", "gemini"]
    assert governor.task_route_preference(governor.TASK_AGENT) == []


def test_matching_is_by_substring_so_renames_survive(monkeypatch):
    """Same rule as ROUTER_ORDER — the route is named 'Groq (llama-3.3-70b)'
    in one place and 'groq' in config."""
    monkeypatch.setenv("ROUTER_ROUTE_CRITIC", "groq")

    assert governor.task_route_rank("Groq (llama-3.3)", governor.TASK_CRITIC) == 0


def test_preference_order_is_the_listed_order(monkeypatch):
    monkeypatch.setenv("ROUTER_ROUTE_SUMMARY", "cohere,groq,gemini")

    ranks = {
        name: governor.task_route_rank(name, governor.TASK_SUMMARY)
        for name in ("Gemini", "Groq", "Cohere")
    }

    assert ranks["Cohere"] < ranks["Groq"] < ranks["Gemini"]


def test_an_unlisted_route_ranks_behind_every_listed_one(monkeypatch):
    monkeypatch.setenv("ROUTER_ROUTE_CRITIC", "groq")

    assert (
        governor.task_route_rank("Pollinations", governor.TASK_CRITIC)
        > governor.task_route_rank("Groq", governor.TASK_CRITIC)
    )


def test_an_unknown_task_name_is_inert(monkeypatch):
    monkeypatch.setenv("ROUTER_ROUTE_NONSENSE", "groq")

    assert governor.task_route_preference("nonsense") == []


# ── The router's sort ──

def _order(routes, task, waits):
    """Run the real ordering `call_ai` uses. Names in, names out.

    Deliberately calls governor.order_routes rather than restating its sort —
    a test that reimplements the logic it is checking passes whenever both
    copies are wrong together, which is exactly when it matters.
    """
    providers = [{"name": name} for name in routes]
    ordered = governor.order_routes(
        providers, task, lambda p: waits.get(p["name"], 0)
    )
    return [p["name"] for p in ordered]


def test_the_preferred_route_goes_first_when_everything_is_free(monkeypatch):
    monkeypatch.setenv("ROUTER_ROUTE_CRITIC", "groq")

    order = _order(["Pollinations", "Gemini", "Groq"], governor.TASK_CRITIC, {})

    assert order[0] == "Groq"


def test_preference_never_costs_a_stall(monkeypatch):
    """INVARIANT: headroom is the outer sort key. A preferred route with no
    token budget left this minute must not make the call wait for it while a
    ready route sits behind it — that trades a better model for a 429."""
    monkeypatch.setenv("ROUTER_ROUTE_CRITIC", "groq")

    order = _order(
        ["Pollinations", "Groq"], governor.TASK_CRITIC, {"Groq": 42.0},
    )

    assert order[0] == "Pollinations"


def test_preference_still_decides_among_equally_ready_routes(monkeypatch):
    monkeypatch.setenv("ROUTER_ROUTE_CRITIC", "gemini")

    order = _order(
        ["Pollinations", "Gemini", "Groq"],
        governor.TASK_CRITIC,
        {"Pollinations": 0, "Gemini": 0, "Groq": 0},
    )

    assert order[0] == "Gemini"


def test_no_route_is_ever_removed(monkeypatch):
    """The preference reorders; it does not filter. A configured route that
    does not exist must not shrink the list."""
    monkeypatch.setenv("ROUTER_ROUTE_CRITIC", "a-route-that-does-not-exist")

    routes = ["Pollinations", "Gemini", "Groq"]
    order = _order(routes, governor.TASK_CRITIC, {})

    assert sorted(order) == sorted(routes)


def test_order_routes_does_not_mutate_its_input():
    """call_ai reassigns the result; a sort in place would reorder PROVIDERS
    itself and let one critic call permanently bias every later call."""
    providers = [{"name": "Pollinations"}, {"name": "Groq"}]
    original = list(providers)

    governor.order_routes(providers, governor.TASK_CRITIC, lambda p: 0)

    assert providers == original


def test_agent_calls_are_unaffected_by_a_critic_preference(monkeypatch):
    monkeypatch.setenv("ROUTER_ROUTE_CRITIC", "groq")

    order = _order(["Pollinations", "Groq"], governor.TASK_AGENT, {})

    assert order == ["Pollinations", "Groq"]


# ── The injection adapter ──

def test_a_two_argument_callable_is_returned_untouched():
    """INVARIANT: the tests inject a scripted AI taking exactly two positional
    arguments. Handing it a keyword would raise, so the tag is only applied to
    callables that can take it."""
    def scripted(messages, system_prompt=None):
        return "ok"

    assert ai_routing.for_task(scripted, governor.TASK_CRITIC) is scripted


def test_a_task_aware_callable_gets_the_tag():
    seen = {}

    def real(messages, system_prompt=None, task="agent"):
        seen["task"] = task
        return "ok"

    ai_routing.for_task(real, governor.TASK_CRITIC)([], "prompt")

    assert seen["task"] == governor.TASK_CRITIC


def test_a_kwargs_callable_gets_the_tag():
    seen = {}

    def flexible(messages, system_prompt=None, **kwargs):
        seen.update(kwargs)
        return "ok"

    ai_routing.for_task(flexible, governor.TASK_SUMMARY)([], "prompt")

    assert seen["task"] == governor.TASK_SUMMARY


def test_an_explicit_task_argument_wins_over_the_tag():
    seen = {}

    def real(messages, system_prompt=None, task="agent"):
        seen["task"] = task
        return "ok"

    ai_routing.for_task(real, governor.TASK_CRITIC)([], "p", task=governor.TASK_SUMMARY)

    assert seen["task"] == governor.TASK_SUMMARY


def test_none_survives():
    assert ai_routing.for_task(None, governor.TASK_CRITIC) is None


def test_a_signatureless_callable_does_not_raise():
    """Builtins have no introspectable signature. Routing as before is the
    correct degradation; raising would take the critic offline."""
    assert ai_routing.for_task(len, governor.TASK_CRITIC) is len


def test_the_adapter_does_not_swallow_a_real_type_error():
    """Catching TypeError instead of inspecting would turn a genuine bug
    inside the AI call into a silent fallback to the untagged path."""
    def real(messages, system_prompt=None, task="agent"):
        raise TypeError("something genuinely wrong inside the call")

    with pytest.raises(TypeError, match="genuinely wrong"):
        ai_routing.for_task(real, governor.TASK_CRITIC)([], "p")


# ── End to end through the critic ──

def test_the_critic_asks_for_the_critic_route():
    import critic

    seen = {}

    def call_ai(messages, system_prompt=None, task="agent"):
        seen["task"] = task
        return "VERDICT: ACCEPT"

    verdict = critic.review("goal", [{"tool": "x", "result": "y"}], "done", call_ai)

    assert verdict.accepted
    assert seen["task"] == governor.TASK_CRITIC


def test_the_critic_still_works_with_a_two_argument_fake():
    """The whole test suite drives the critic this way; the adapter must not
    have quietly broken that contract."""
    import critic

    calls = []

    def scripted(messages, system_prompt=None):
        calls.append(messages)
        return "VERDICT: REVISE\nREASON: the file was never written"

    verdict = critic.review("goal", [{"tool": "x", "result": "y"}], "done", scripted)

    assert not verdict.accepted
    assert len(calls) == 1
