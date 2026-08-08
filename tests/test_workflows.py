"""
The recurring jobs, and the constraints their goals have to respect.

A workflow goal is text that will be handed to an unattended run days from
now, with nobody present to correct it. The things that make such a goal fail
are not syntax — they are a schedule spec that does not parse, a tool that
does not exist, an instruction to use a tool the run engine blocks, or a
missing token discovered at 9am instead of at scheduling time. Each is
checked here, because none of them shows up until the run fires.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import governor
import memory
import tools
import triggers
import workflows


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "memory.db")
    yield


# ── The presets themselves ──

def test_every_schedule_spec_parses():
    """An unparseable spec fails at scheduling time, which is the good case —
    but only if someone runs it. This is that someone."""
    for name, preset in workflows.PRESETS.items():
        triggers.next_run_after(preset["when"])  # raises ValueError if bad


def test_every_goal_stands_alone():
    """
    A run starts with no conversation context beyond this text, so a goal that
    says "the repo we discussed" has already failed.
    """
    for name, preset in workflows.PRESETS.items():
        goal = preset["goal"]
        assert len(goal) > 200, f"{name}: too thin to act on unattended"
        lowered = goal.lower()
        for phrase in ("as discussed", "we talked about", "the one you mentioned",
                       "as above", "like last time"):
            assert phrase not in lowered, f"{name}: leans on absent context"


def test_every_tool_named_in_a_goal_exists():
    """
    A goal naming a tool that was renamed or never existed sends the agent
    hunting through its registry at 9am, spending calls on a typo.
    """
    for name, preset in workflows.PRESETS.items():
        for word in preset["goal"].replace("(", " ").replace(",", " ").split():
            candidate = word.strip(".:;'\"")
            # only check things that look like a tool call in the prose
            if candidate in tools.TOOLS or candidate.endswith("_tool"):
                assert candidate in tools.TOOLS, f"{name}: unknown tool {candidate}"


def test_no_goal_asks_for_a_tool_unattended_runs_block():
    """
    Unattended runs block the tools that spawn more autonomous work, and
    EXTERNAL tools park the run waiting for an approval nobody is there to
    give. A goal built around one of those is a goal that cannot finish.
    """
    blocked = tools.UNATTENDED_BLOCKED_TOOLS
    for name, preset in workflows.PRESETS.items():
        for tool_name in blocked:
            assert tool_name not in preset["goal"], (
                f"{name}: goal calls {tool_name}, which unattended runs block"
            )


def test_goals_that_touch_external_tools_say_to_stop_instead():
    """
    Every goal must name its boundary. Reaching an EXTERNAL tool unattended
    parks the run on an approval queue; the goal should have said "report
    this" long before that.
    """
    for name, preset in workflows.PRESETS.items():
        lowered = preset["goal"].lower()
        assert any(
            phrase in lowered
            for phrase in ("do not push", "do not fix", "do not restart",
                           "report only", "leave it to a human", "needs a human",
                           "should look at", "needs running", "save nothing")
        ), f"{name}: goal never says where to stop"


def test_presets_declare_what_they_need():
    """A workflow reading GitHub without a token is a scheduled failure."""
    for name, preset in workflows.PRESETS.items():
        assert isinstance(preset["needs"], list)
        if "github" in preset["goal"].lower():
            assert "GITHUB_TOKEN" in preset["needs"], f"{name}: reads GitHub, needs token"


# ── Starting one ──

def test_missing_config_is_refused_at_scheduling_time(monkeypatch):
    """
    Better a refusal now than a run reporting a missing token at 9am, when
    the connection back to this moment is gone.
    """
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_DEFAULT_OWNER", raising=False)
    monkeypatch.delenv("GITHUB_DEFAULT_REPO", raising=False)

    result = workflows.start("repo-review", owner_user_id="U1")

    assert result.startswith("❌")
    assert "GITHUB_TOKEN" in result
    assert triggers.list_schedules() == [], "nothing should have been scheduled"


def test_a_configured_workflow_schedules(monkeypatch):
    for key in ("GITHUB_TOKEN", "GITHUB_DEFAULT_OWNER", "GITHUB_DEFAULT_REPO"):
        monkeypatch.setenv(key, "x")

    result = workflows.start("repo-review", owner_user_id="U1", channel="C1")

    assert result.startswith("✅")
    scheduled = triggers.list_schedules()
    assert [s["name"] for s in scheduled] == ["repo-review"]
    assert scheduled[0]["goal"] == workflows.PRESETS["repo-review"]["goal"]


def test_a_workflow_with_no_requirements_needs_no_config():
    result = workflows.start("ops-watch", owner_user_id="U1")
    assert result.startswith("✅")


def test_an_unknown_name_is_refused_and_lists_the_real_ones():
    result = workflows.start("nope", owner_user_id="U1")
    assert result.startswith("❌")
    assert "repo-review" in result


def test_the_schedule_can_be_overridden():
    workflows.start("ops-watch", owner_user_id="U1", when="daily 07:00")
    assert triggers.list_schedules()[0]["spec"] == "daily 07:00"


def test_a_bad_override_is_reported_not_raised():
    result = workflows.start("ops-watch", owner_user_id="U1", when="whenever")
    assert result.startswith("❌")
    assert triggers.list_schedules() == []


def test_describe_lists_every_preset():
    text = workflows.describe()
    for name in workflows.PRESETS:
        assert name in text


# ── The tools the workflows depend on ──

def test_the_pull_request_tools_exist_and_are_read_tier():
    """
    repo-review cannot work without these: github_list_issues filters PRs out
    entirely, so listing them needed its own tool.
    """
    for name in ("github_list_pull_requests", "github_pr_status"):
        assert name in tools.TOOLS
        assert governor.tier_of(name) == governor.READ
