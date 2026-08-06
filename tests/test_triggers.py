"""
Tests for the trigger layer: schedule specs, the scheduler tick, and the
plan sweeper that resumes unfinished work.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memory
import runner
import triggers


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "memory.db")
    yield


def ts(text: str) -> float:
    """Local-time 'YYYY-MM-DD HH:MM' → unix seconds."""
    return time.mktime(time.strptime(text, "%Y-%m-%d %H:%M"))


def fmt(when: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(when))


# ── Spec parsing ──

@pytest.mark.parametrize("spec,expected", [
    ("every 15m", ("interval", 900)),
    ("every 2h", ("interval", 7200)),
    ("every 1d", ("interval", 86400)),
    ("every 30 minutes", ("interval", 1800)),
    ("hourly", ("cron", "0 * * * *")),
    ("daily 09:00", ("cron", "0 9 * * *")),
    ("daily 17:30", ("cron", "30 17 * * *")),
    ("weekly mon 08:15", ("cron", "15 8 * * 1")),
    ("0 9 * * 1-5", ("cron", "0 9 * * 1-5")),
    ("cron 0 9 * * 1-5", ("cron", "0 9 * * 1-5")),
])
def test_parse_spec(spec, expected):
    assert triggers.parse_spec(spec) == expected


@pytest.mark.parametrize("spec", [
    "", "every", "every banana", "every 30s", "daily 25:00",
    "weekly funday 09:00", "0 9 * *", "nonsense", "0 9 * * abc",
])
def test_bad_specs_are_rejected_with_help(spec):
    with pytest.raises(ValueError):
        triggers.parse_spec(spec)


def test_sub_minute_intervals_rejected():
    with pytest.raises(ValueError, match="Minimum interval"):
        triggers.parse_spec("every 30s")


# ── next_run_after ──

def test_interval_next_run_is_relative():
    base = ts("2026-03-04 10:00")
    assert triggers.next_run_after("every 15m", base) == base + 900


def test_daily_next_run_is_the_next_matching_clock_time():
    assert fmt(triggers.next_run_after("daily 09:00", ts("2026-03-04 08:00"))) == "2026-03-04 09:00"
    # Already past today → tomorrow.
    assert fmt(triggers.next_run_after("daily 09:00", ts("2026-03-04 09:30"))) == "2026-03-05 09:00"


def test_weekday_cron_skips_the_weekend():
    # 2026-03-07 is a Saturday; the next weekday 9am is Monday the 9th.
    assert fmt(triggers.next_run_after("0 9 * * 1-5", ts("2026-03-07 12:00"))) == "2026-03-09 09:00"


def test_weekly_lands_on_the_named_day():
    when = triggers.next_run_after("weekly mon 08:15", ts("2026-03-04 12:00"))
    assert fmt(when) == "2026-03-09 08:15"
    assert time.localtime(when).tm_wday == 0  # Monday


def test_step_syntax_in_cron():
    assert fmt(triggers.next_run_after("*/15 * * * *", ts("2026-03-04 10:02"))) == "2026-03-04 10:15"


def test_sunday_is_both_0_and_7():
    # 2026-03-04 is a Wednesday; next Sunday is the 8th.
    assert fmt(triggers.next_run_after("0 6 * * 0", ts("2026-03-04 10:00"))) == "2026-03-08 06:00"
    assert fmt(triggers.next_run_after("0 6 * * 7", ts("2026-03-04 10:00"))) == "2026-03-08 06:00"


def test_impossible_date_raises_rather_than_hanging():
    with pytest.raises(ValueError, match="never matches"):
        triggers.next_run_after("0 9 30 2 *")  # February 30th


# ── Schedule CRUD ──

def test_add_and_list_schedule():
    triggers.add_schedule("standup", "daily 09:00", "Summarize open PRs", owner_user_id="U1")
    scheds = triggers.list_schedules()
    assert len(scheds) == 1
    assert scheds[0]["name"] == "standup"
    assert scheds[0]["enabled"] == 1
    assert scheds[0]["next_run"] > time.time()


def test_adding_the_same_name_replaces_it():
    triggers.add_schedule("standup", "daily 09:00", "old goal")
    triggers.add_schedule("standup", "hourly", "new goal")
    scheds = triggers.list_schedules()
    assert len(scheds) == 1
    assert scheds[0]["spec"] == "hourly"
    assert scheds[0]["goal"] == "new goal"


def test_bad_spec_never_creates_a_schedule():
    with pytest.raises(ValueError):
        triggers.add_schedule("broken", "every fortnight", "do a thing")
    assert triggers.list_schedules() == []


def test_cancel_schedule_by_name():
    triggers.add_schedule("standup", "hourly", "goal")
    assert triggers.cancel_schedule("standup") is True
    assert triggers.cancel_schedule("standup") is False
    assert triggers.list_schedules() == []


# ── Firing ──

def test_due_schedule_queues_a_run_and_advances():
    triggers.add_schedule("standup", "every 1h", "Check the repo", owner_user_id="U1",
                          channel="C1")
    triggers._set_schedule_fields(1, next_run=time.time() - 5)

    run_ids = triggers.fire_due_schedules()
    assert len(run_ids) == 1

    run = runner.get_run(run_ids[0])
    assert run["source"] == "schedule"
    assert run["goal"] == "Check the repo"
    assert run["channel"] == "C1"
    assert run["owner_user_id"] == "U1"
    assert run["unattended"] == 1
    assert run["label"] == "standup"

    sched = triggers.list_schedules()[0]
    assert sched["next_run"] > time.time()
    assert sched["run_count"] == 1


def test_schedule_not_yet_due_does_nothing():
    triggers.add_schedule("later", "daily 09:00", "goal")
    assert triggers.fire_due_schedules() == []


def test_disabled_schedule_never_fires():
    triggers.add_schedule("off", "every 1h", "goal")
    triggers._set_schedule_fields(1, enabled=0, next_run=time.time() - 5)
    assert triggers.fire_due_schedules() == []


def test_schedule_still_advances_when_the_run_budget_refuses(monkeypatch):
    """A rejected run must not leave the schedule due forever, re-firing every tick."""
    monkeypatch.setenv("RUN_DAILY_LIMIT", "1")
    runner.enqueue_run("uses up the budget", source="schedule")

    triggers.add_schedule("blocked", "every 1h", "goal")
    triggers._set_schedule_fields(1, next_run=time.time() - 5)

    assert triggers.fire_due_schedules() == []
    assert triggers.list_schedules()[0]["next_run"] > time.time()


def test_schedule_does_not_stack_runs_on_top_of_itself():
    """A slow run must delay the next fire, not queue a backlog behind it."""
    triggers.add_schedule("slow", "every 1h", "goal", channel="C1")
    triggers._set_schedule_fields(1, next_run=time.time() - 5)
    assert len(triggers.fire_due_schedules()) == 1

    # Its run is still going; the next due tick must not queue a second one.
    triggers._set_schedule_fields(1, next_run=time.time() - 5)
    assert triggers.fire_due_schedules() == []
    assert len([r for r in runner.list_runs(limit=10) if r["source"] == "schedule"]) == 1
    # …and the schedule still advanced, so it isn't stuck due forever.
    assert triggers.list_schedules()[0]["next_run"] > time.time()


def test_schedule_fires_again_once_its_previous_run_finished():
    triggers.add_schedule("slow", "every 1h", "goal", channel="C1")
    triggers._set_schedule_fields(1, next_run=time.time() - 5)
    first = triggers.fire_due_schedules()[0]

    runner._update_run(first, status="done")
    triggers._set_schedule_fields(1, next_run=time.time() - 5)
    assert len(triggers.fire_due_schedules()) == 1


def test_overlap_can_be_allowed_explicitly(monkeypatch):
    monkeypatch.setenv("SCHEDULE_ALLOW_OVERLAP", "true")
    triggers.add_schedule("parallel", "every 1h", "goal", channel="C1")
    triggers._set_schedule_fields(1, next_run=time.time() - 5)
    triggers.fire_due_schedules()
    triggers._set_schedule_fields(1, next_run=time.time() - 5)
    assert len(triggers.fire_due_schedules()) == 1


def test_a_different_schedule_is_not_blocked_by_someone_elses_run():
    triggers.add_schedule("one", "every 1h", "goal a", channel="C1")
    triggers.add_schedule("two", "every 1h", "goal b", channel="C1")
    for sid in (1, 2):
        triggers._set_schedule_fields(sid, next_run=time.time() - 5)
    assert len(triggers.fire_due_schedules()) == 2


# ── Plan sweeper ──

def make_stale_plan(conv_key="C1:main", user_id="U1", age_seconds=3600, steps=None):
    """A plan with open steps, last touched `age_seconds` ago."""
    memory.create_plan(conv_key, user_id, steps or ["step one", "step two"])
    memory.update_task_status(conv_key, 1, "done")
    old = time.time() - age_seconds
    conn = memory.get_db()
    try:
        conn.execute("UPDATE tasks SET updated = ? WHERE conv_key = ?", (old, conv_key))
        conn.commit()
    finally:
        conn.close()


def test_quiet_unfinished_plan_is_resumed():
    make_stale_plan()
    run_ids = triggers.sweep_stale_plans()

    assert len(run_ids) == 1
    run = runner.get_run(run_ids[0])
    assert run["source"] == "plan_resume"
    assert run["conv_key"] == "C1:main"
    assert run["owner_user_id"] == "U1"
    assert run["channel"] == "C1"
    assert "step two" in run["goal"]


def test_finished_plan_is_left_alone():
    memory.create_plan("C1:main", "U1", ["only step"])
    memory.update_task_status("C1:main", 1, "done")
    assert triggers.sweep_stale_plans() == []


def test_recent_plan_is_left_alone():
    make_stale_plan(age_seconds=10)
    assert triggers.sweep_stale_plans() == []


def test_active_conversation_is_never_interrupted():
    """The human is still typing — resuming here would talk over them."""
    make_stale_plan()
    memory.add_message("C1:main", "user", "wait, hold on")
    assert triggers.sweep_stale_plans() == []


def test_plan_with_a_run_already_going_is_not_double_queued():
    make_stale_plan()
    assert len(triggers.sweep_stale_plans()) == 1
    assert triggers.sweep_stale_plans() == []  # first run is still active


def test_resume_cap_stops_an_endless_loop(monkeypatch):
    monkeypatch.setenv("PLAN_MAX_RESUMES", "2")
    make_stale_plan()

    for _ in range(4):
        for run in runner.list_runs(limit=20):
            if run["status"] in runner.ACTIVE_STATUSES:
                runner._update_run(run["id"], status="done")
        triggers.sweep_stale_plans()

    resumes = [r for r in runner.list_runs(limit=50) if r["source"] == "plan_resume"]
    assert len(resumes) == 2


def test_auto_resume_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("PLAN_AUTO_RESUME", "false")
    make_stale_plan()
    assert triggers.sweep_stale_plans() == []


# ── Tick ──

def test_tick_fires_schedules_and_sweeps_plans():
    triggers.add_schedule("due-now", "every 1h", "scheduled goal")
    triggers._set_schedule_fields(1, next_run=time.time() - 5)
    make_stale_plan(conv_key="C2:main")

    result = triggers.tick()
    assert len(result["schedules_fired"]) == 1
    assert len(result["plans_resumed"]) == 1


def test_tick_survives_a_broken_schedule(monkeypatch):
    def explode(*a, **kw):
        raise RuntimeError("db on fire")

    monkeypatch.setattr(triggers, "fire_due_schedules", explode)
    make_stale_plan()

    result = triggers.tick()  # must not raise — the ticker thread has to survive
    assert result["schedules_fired"] == []
    assert len(result["plans_resumed"]) == 1
