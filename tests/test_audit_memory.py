"""
Phase 3.9 — memory: scoping, isolation, /clear, and growth.

Invariants under test:
  • Thread history is keyed by conv_key and invisible to other threads.
  • category='decision' is durable project memory and reaches a NEW thread;
    category='fact' is ambient and capped.
  • One user's memory never reaches another user.
  • /clear does what its confirmation message says.
  • Injected memory cannot crowd the task out of the context budget.

`xfail(strict=True)` marks a CONFIRMED DEFECT — see it fail with
`pytest --runxfail`.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_support import Say, audit_env, slack_bot, slash  # noqa: F401

import agent
import memory
import runner


# ── scoping ──

def test_thread_history_is_keyed_by_conv_key(audit_env):
    memory.add_message("C1:100", "user", "alpha thread secret")
    memory.add_message("C1:200", "user", "beta thread secret")
    assert [m["content"] for m in memory.get_history("C1:100", limit=10)] == \
        ["alpha thread secret"]


def test_decisions_reach_a_brand_new_thread(audit_env):
    """This is the point of durable memory: a new thread still knows the call."""
    memory.add_fact("U_OWNER", "we chose SQLite over Postgres for the 1 GB box",
                    category="decision")
    prompt = agent.get_agent_system_prompt("base", user_facts=memory.get_facts("U_OWNER"))
    assert "SQLite over Postgres" in prompt
    assert "PROJECT MEMORY" in prompt


def test_plain_facts_are_capped_and_not_load_bearing(audit_env):
    """Ambient facts must not grow the prompt without bound, and must be
    rendered separately from decisions so they cannot masquerade as one."""
    for i in range(40):
        memory.add_fact("U_OWNER", f"ambient preference {i}", category="fact")
    facts = memory.get_facts("U_OWNER")
    assert len(facts["recent"]) == 10, "ambient facts are not capped"
    assert facts["durable"] == []
    prompt = agent.get_agent_system_prompt("base", user_facts=facts)
    assert "ambient preference 39" in prompt and "ambient preference 5" not in prompt


def test_decisions_are_capped_too(audit_env):
    for i in range(60):
        memory.add_fact("U_OWNER", f"decision {i}", category="decision")
    assert len(memory.get_facts("U_OWNER")["durable"]) == 40


def test_one_users_decisions_never_reach_another(audit_env):
    memory.add_fact("U_ALICE", "alice's salary band is X", category="decision")
    other = memory.get_facts("U_BOB")
    assert other == {"durable": [], "recent": []}
    assert "salary band" not in agent.get_agent_system_prompt("base", user_facts=other)


@pytest.mark.xfail(strict=True, reason=(
    "FINDING I1 (HIGH) — cross-thread recall is not scoped by user. "
    "memory.search_all_relevant (memory.py:182) takes only a query and an "
    "excluded conv_key, and bot.process_message (bot.py:850) feeds its results "
    "into the prompt for whoever is talking. One user's private DM with the bot "
    "is therefore quotable into another user's public-channel thread whenever "
    "the keywords overlap. Same root cause as FINDING F2 (memory_search); this "
    "one needs no tool call and no injection — it happens on an ordinary "
    "message."))
def test_cross_thread_recall_does_not_cross_users(audit_env):
    memory.add_message("D_ALICE:1", "user",
                       "confidential: my severance package is 250000 GBP")
    hits = memory.search_all_relevant("severance package", exclude_conv_key="C_PUBLIC:1")
    assert not any("250000" in h["content"] for h in hits), (
        "another user's DM is recallable into a stranger's thread")


# ── /clear ──

def test_clear_removes_the_channels_thread_history(audit_env, slack_bot):
    bot, registry = slack_bot
    memory.add_message("C_CHAN:1", "user", "channel chatter")
    slash(registry, "/clear", user="U_OWNER", channel="C_CHAN")
    assert memory.get_history("C_CHAN:1", limit=10) == []


def test_clear_does_not_touch_another_channel(audit_env, slack_bot):
    bot, registry = slack_bot
    memory.add_message("C_OTHER:1", "user", "other channel")
    slash(registry, "/clear", user="U_OWNER", channel="C_CHAN")
    assert len(memory.get_history("C_OTHER:1", limit=10)) == 1


@pytest.mark.xfail(strict=True, reason=(
    "FINDING I2 (MEDIUM) — /clear says 'Memory cleared!' but deletes rows from "
    "`conversations` only (bot.py:1058-1072). thread_summaries, facts "
    "(including decisions), the concept graph and run transcripts all survive, "
    "and cross-thread recall + rolling summaries put that content straight back "
    "into the next reply. A user who clears a channel because something "
    "sensitive was said has not removed it. README calls the command 'Reset "
    "conversation memory', which is nearer the truth than the confirmation "
    "message is."))
def test_clear_clears_what_it_claims(audit_env, slack_bot):
    bot, registry = slack_bot
    memory.add_message("C_CHAN:1", "user", "my card number is 4111 1111 1111 1111")
    memory.save_thread_summary("C_CHAN:1", "U_OWNER",
                               "user shared card number 4111 1111 1111 1111", 12)
    memory.add_fact("U_OWNER", "card on file ends 1111", category="decision")

    say, _ = slash(registry, "/clear", user="U_OWNER", channel="C_CHAN")
    assert "cleared" in (say.last or "").lower()

    leftovers = memory.search_all_relevant("card number")
    assert not leftovers, f"still recallable after /clear: {leftovers}"
    assert memory.get_facts("U_OWNER")["durable"] == []


# ── growth ──

def test_injected_memory_cannot_crowd_out_the_task(audit_env):
    """The number you asked for: how much prompt do 40 decisions buy?

    get_facts caps durable memory at 40 entries; the system prompt renders all
    of them. At a realistic ~120 chars per decision that is ~5k chars against a
    24k-char transcript budget — 20%, on top of a ~35k-char base prompt.
    """
    base = agent.get_agent_system_prompt("base", user_facts={"durable": [], "recent": []})
    for i in range(60):                      # more than the cap, deliberately
        memory.add_fact("U_OWNER", f"decision {i}: " + "x" * 110, category="decision")
    loaded = agent.get_agent_system_prompt("base", user_facts=memory.get_facts("U_OWNER"))

    injected = len(loaded) - len(base)
    budget = runner.context_limit_chars()
    assert injected < budget * 0.5, (
        f"memory injection is {injected} chars against a {budget}-char transcript "
        f"budget — it would crowd out the task")
    # Recorded for the report, not asserted as a threshold:
    print(f"\nbase prompt {len(base):,} chars; 40 capped decisions add {injected:,} "
          f"chars ({injected / budget:.0%} of the {budget:,}-char transcript budget)")


def test_the_cap_is_what_bounds_growth_not_the_row_count(audit_env):
    """1000 decisions must cost the same prompt as 40."""
    for i in range(1000):
        memory.add_fact("U_OWNER", f"decision {i}", category="decision")
    prompt = agent.get_agent_system_prompt("base", user_facts=memory.get_facts("U_OWNER"))
    assert prompt.count("decision ") == 40


def test_concept_graph_stays_small_as_the_db_grows(audit_env):
    """NetworkX is in-process on a 1 GB box; the graph must not scale with
    every message ever stored."""
    import resource

    import concept_graph
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for i in range(300):
        concept_graph.extract_and_store(
            f"Project Apollo{i} uses Python and runs on Ubuntu with SQLite.", f"C{i}:1")
    stats = concept_graph.get_graph_stats()
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    growth_mb = (after - before) / 1024
    print(f"\nconcept graph after 300 extractions: {stats['entities']} entities, "
          f"{stats['edges']} edges, peak RSS +{growth_mb:.1f} MB")
    assert growth_mb < 100, f"graph grew {growth_mb:.1f} MB for 300 messages"
