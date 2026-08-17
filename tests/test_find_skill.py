"""
Skill retrieval.

The playbooks in `skills/` shipped with the repo but were unreachable: the
system prompt said "full references live in skills/coding-practices/ — read
the relevant file", while `read_file` basenames into ~/agent_workspace and
`repo_read_file` resolves under the clone directory. Neither can open the
bot's own checkout. The skills README named this gap and anticipated "a future
tool that can read this repo directly"; find_skill is it.

Two properties carry the weight. Retrieval must beat inlining — 289 KB of
playbooks against a system prompt re-sent on every call in a ten-iteration
loop. And a miss must return *nothing*: the model reads a returned playbook as
instruction, so the least-bad match is worse than no match.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import skills_index
import tools


@pytest.fixture(autouse=True)
def _fresh_index():
    skills_index.reload()
    yield
    skills_index.reload()


# ── The library is actually reachable now ──

def test_the_shipped_playbooks_are_found():
    """The regression this fixes: they were on disk and no tool could open
    them."""
    names = [s.name for s in skills_index.all_skills()]

    assert len(names) >= 20
    assert "test-driven-development" in names


def test_frontmatter_is_parsed():
    skill = skills_index.get("test-driven-development")

    assert skill.description
    assert "use when" in skill.description.lower()
    assert skill.body.startswith("#")


def test_the_prompt_no_longer_points_at_an_unreachable_path():
    """INVARIANT: an instruction the model structurally cannot follow is worse
    than no instruction — it burns steps and then invents a substitute."""
    import agent

    prompt = agent.get_agent_system_prompt("base", [])

    assert "find_skill" in prompt
    assert "read the relevant file" not in prompt


# ── Ranking ──

@pytest.mark.parametrize("query,expected", [
    ("I need to fix a bug and prove it stays fixed", "test-driven-development"),
    ("review someone else's pull request", "code-review-and-quality"),
    ("harden this against attackers", "security-and-hardening"),
    ("write a spec before I start coding", "spec-driven-development"),
])
def test_a_task_description_finds_the_right_playbook(query, expected):
    assert skills_index.search(query)[0].name == expected


def test_an_ambiguous_query_still_surfaces_the_plausible_playbooks():
    """"Break a big feature into steps" is genuinely both planning and
    incremental implementation. Ranking one first is fine; dropping the other
    is not, which is why the tool names the runners-up."""
    names = [s.name for s in skills_index.search("break a big feature into steps")]

    assert "planning-and-task-breakdown" in names
    assert "incremental-implementation" in names


def test_a_body_only_match_does_not_qualify():
    """INVARIANT: a 300-line document about software contains almost any
    software-adjacent word somewhere. Body matches break ties between real
    candidates; they must not create one."""
    joke_terms = skills_index._terms("tell me a joke about penguins")
    scored = [s for s in skills_index.all_skills() if s.score(joke_terms) > 0]

    assert scored == []


def test_a_query_no_playbook_covers_returns_nothing():
    """INVARIANT: the threshold is what keeps this honest. Handing the model
    an irrelevant playbook is worse than handing it none."""
    assert skills_index.search("what is the weather in paris tomorrow") == []
    assert skills_index.search("tell me a joke about penguins") == []


@pytest.mark.parametrize("query", [
    "what is the weather in paris tomorrow",
    "tell me a joke about penguins",
    "who won the world cup in 1998",
    "translate this sentence into german",
    "what time is my meeting",
    "summarise this article for me",
    "remind me to call the dentist",
])
def test_everyday_requests_match_nothing(query):
    """The tool sits in a general-purpose chat bot, so most turns are not
    software engineering at all. Every one of these returning a playbook would
    hand the model instructions for a task it isn't doing."""
    assert skills_index.search(query) == []


def test_stopwords_do_not_carry_a_match():
    """Without a stoplist, 'the code' scores every playbook in a corpus about
    software equally and the ranking becomes noise."""
    assert skills_index.search("the code that you use for this") == []


def test_an_empty_query_matches_nothing_rather_than_everything():
    assert skills_index.search("") == []
    assert skills_index.search("   ") == []


def test_a_single_overlapping_word_is_not_a_match():
    """INVARIANT: one word in common is coincidence, two is a topic. "what
    time is my meeting" overlapped exactly one description term and outscored
    every genuine query — IDF cannot fix that, because "meeting" really is a
    rare word that really does appear in one description."""
    assert skills_index.search("what time is my meeting") == []

    # …while every real query overlaps at least two.
    for query in (
        "I need to fix a bug and prove it stays fixed",
        "harden this against attackers",
        "review someone else's pull request",
    ):
        assert skills_index.search(query), query


def test_generic_terms_are_discounted_against_the_corpus():
    """A term in most descriptions identifies nothing; a term in one identifies
    it. Derived from the corpus rather than curated, because no hand-written
    stoplist anticipates which words are generic across *these* documents."""
    common = max(skills_index._document_frequency().items(), key=lambda kv: kv[1])[0]

    assert skills_index._idf(common) < skills_index._idf("attackers")


def test_a_name_match_is_not_discounted():
    """A name is an identity, not a mention. Discounting it by corpus
    frequency ranked planning-and-task-breakdown above
    spec-driven-development for "write a spec before I start coding"."""
    assert skills_index.search("write a spec before I start coding")[0].name == (
        "spec-driven-development"
    )


def test_score_is_normalised_by_query_length():
    """Otherwise a long question always outranks a short one and the
    threshold means something different for each."""
    short = skills_index.search("test driven development")
    long_ = skills_index.search(
        "I would like to do test driven development on this particular "
        "project because writing tests first seems like a good idea"
    )

    assert short and long_
    assert short[0].name == long_[0].name == "test-driven-development"


# ── Lookup by name ──

def test_name_lookup_tolerates_how_a_model_writes_it():
    for spelling in (
        "test-driven-development", "test driven development",
        "Test Driven Development", "TestDrivenDevelopment",
    ):
        assert skills_index.get(spelling).name == "test-driven-development"


def test_an_unknown_name_is_none():
    assert skills_index.get("astrology-driven-development") is None
    assert skills_index.get("") is None


# ── Rendering ──

def test_a_long_playbook_is_cut_on_a_heading_boundary():
    skill = max(skills_index.all_skills(), key=lambda s: len(s.body))

    rendered = skills_index.render(skill, max_chars=1200)

    assert "truncated" in rendered
    assert len(rendered) < len(skill.body)
    # The cut lands before a heading, not mid-sentence.
    body_tail = rendered.split("…(truncated")[0].rstrip()
    assert not body_tail.endswith(",")


def test_a_short_playbook_is_returned_whole():
    skill = min(skills_index.all_skills(), key=lambda s: len(s.body))

    assert "truncated" not in skills_index.render(skill, max_chars=100_000)


# ── The tool ──

def test_the_tool_returns_the_playbook_body():
    out = tools.find_skill("write a failing test before the fix")

    assert "Test-Driven Development" in out


def test_the_tool_lists_the_index_with_no_query():
    out = tools.find_skill()

    assert "test-driven-development" in out
    assert out.count("•") >= 20


def test_the_tool_names_alternatives_when_several_match():
    out = tools.find_skill("debug a failing test in code review")

    assert "Also possibly relevant" in out


def test_the_tool_reads_one_by_name():
    out = tools.find_skill(name="git-workflow-and-versioning")

    assert "git" in out.lower()


def test_an_unknown_name_lists_what_exists():
    out = tools.find_skill(name="nonsense-driven-development")

    assert out.startswith("❌")
    assert "test-driven-development" in out


def test_a_miss_tells_the_model_to_use_its_own_judgment():
    out = tools.find_skill("what is the capital of france")

    assert "No playbook covers" in out
    assert "own judgment" in out


def test_find_skill_is_a_read(monkeypatch):
    import governor

    assert governor.tier_of("find_skill") == governor.READ


def test_find_skill_is_not_owner_only():
    """It reads files shipped with the code and has no effects. Gating it
    would train people to expect refusals from harmless tools."""
    assert "find_skill" not in tools.OWNER_ONLY_TOOLS


def test_a_missing_skills_directory_is_not_an_error(monkeypatch):
    """A deployment that did not vendor the playbooks must degrade to 'none
    installed', not crash the tool registry."""
    monkeypatch.setattr(skills_index, "SKILLS_DIR", "/nonexistent/skills")
    skills_index.reload()

    assert skills_index.all_skills() == []
    assert "No skill playbooks" in tools.find_skill()
    assert skills_index.search("anything") == []
