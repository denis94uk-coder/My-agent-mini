"""
Phase 2.8 — schedule parsing, firing, and the timezone assumption.

Invariants under test:
  • A spec either parses into something that can fire, or is rejected loudly.
    A spec that is accepted and then never fires is the worst outcome.
  • Firing does not stack runs, and a schedule with an unusable spec is
    disabled rather than retried every tick.
  • Duplicate names replace; cancelling an unknown name is False, not an error.

`xfail(strict=True)` marks a CONFIRMED DEFECT — see it fail with
`pytest --runxfail`.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_support import audit_env, run_id_of  # noqa: F401

import runner
import triggers


# ── parsing ──

@pytest.mark.parametrize("spec,expected", [
    ("every 15m", ("interval", 900)),
    ("every 2h", ("interval", 7200)),
    ("every 1d", ("interval", 86400)),
    ("every 15 minutes", ("interval", 900)),
    ("hourly", ("cron", "0 * * * *")),
    ("daily 09:00", ("cron", "0 9 * * *")),
    ("daily 9", ("cron", "0 9 * * *")),
    ("weekly mon 08:15", ("cron", "15 8 * * 1")),
    ("0 9 * * 1-5", ("cron", "0 9 * * 1-5")),
    ("* * * * *", ("cron", "* * * * *")),
    ("cron 0 9 * * *", ("cron", "0 9 * * *")),
    ("  DAILY 09:00  ", ("cron", "0 9 * * *")),
])
def test_specs_that_must_parse(audit_env, spec, expected):
    assert triggers.parse_spec(spec) == expected


@pytest.mark.parametrize("spec", [
    "", "   ", "every", "every banana", "every 0m", "every 30s",
    "daily 25:00", "daily 09:61", "daily noon",
    "weekly funday 08:15", "weekly", "0 9 * *", "0 9 * * * *",
    "cron notacron x y z", "x" * 400,
])
def test_specs_that_must_be_rejected(audit_env, spec):
    with pytest.raises(ValueError):
        triggers.parse_spec(spec)


@pytest.mark.xfail(strict=True, reason=(
    "FINDING E1 (LOW) — the interval parser discards everything that is not a "
    "digit or a letter (triggers.py:109), so `every -5m` becomes `every 5m` "
    "and `every 1.5h` becomes `every 15h`. A negative or fractional interval "
    "is a typo the user should hear about; instead the schedule is created "
    "and confirmed at a cadence they did not ask for."))
@pytest.mark.parametrize("spec", ["every -5m", "every 1.5h", "every +2h"])
def test_malformed_intervals_are_rejected(audit_env, spec):
    with pytest.raises(ValueError):
        triggers.parse_spec(spec)


@pytest.mark.parametrize("spec", ["99 99 * * *", "0 25 * * *", "0 9 32 13 *",
                                 "0 9 * 13 *", "99 9 * * *", "0 9 * * 9",
                                 "30-10 9 * * *"])
def test_out_of_range_cron_fields_are_rejected(audit_env, spec):
    """FIXED (was FINDING E2): every cron field is range-checked, so a spec that
    could never fire is refused at creation instead of listed as active."""
    with pytest.raises(ValueError):
        triggers.parse_spec(spec)


def test_an_impossible_date_fails_loudly_rather_than_hanging(audit_env):
    with pytest.raises(ValueError):
        triggers.next_run_after("0 9 30 2 *")      # Feb 30th


# ── timezone / DST ──

@pytest.fixture
def london(monkeypatch):
    monkeypatch.setenv("TZ", "Europe/London")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def _fires(spec, start, count):
    out, t = [], start
    for _ in range(count):
        t = triggers.next_run_after(spec, t)
        out.append(time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(t)))
    return out


def test_a_daytime_schedule_is_unaffected_by_dst(audit_env, london):
    start = time.mktime((2026, 3, 28, 23, 0, 0, 0, 0, -1))
    assert _fires("daily 09:00", start, 3) == [
        "2026-03-29 09:00 BST", "2026-03-30 09:00 BST", "2026-03-31 09:00 BST"]


def test_a_schedule_in_the_dst_window_fires_exactly_once_a_day(audit_env, london):
    """FIXED (was FINDING E3, autumn half): on the fall-back day 01:30 happens
    twice, an hour apart, and the schedule used to fire on both — a duplicate
    deploy or PR nobody asked for. next_run_after now refuses to return the
    same local wall-clock minute it just fired on."""
    autumn = _fires("daily 01:30", time.mktime((2026, 10, 24, 23, 0, 0, 0, 0, -1)), 3)
    assert len({f.split()[0] for f in autumn}) == len(autumn), f"duplicate day: {autumn}"


def test_the_spring_forward_gap_skips_a_day_rather_than_doubling_up(audit_env, london):
    """ACCEPTED BEHAVIOUR (residual of FINDING E3). 01:30 does not exist on the
    spring-forward day, so that schedule moves to the next day rather than
    firing at a substituted hour. Recorded because it is a real surprise for a
    once-a-day job — and because the safe direction to fail is 'not twice'."""
    spring = _fires("daily 01:30", time.mktime((2026, 3, 28, 23, 0, 0, 0, 0, -1)), 2)
    assert spring[0].startswith("2026-03-30"), spring
    assert not any(f.startswith("2026-03-29") for f in spring)


# ── CRUD ──

def test_duplicate_name_replaces_rather_than_duplicating(audit_env):
    triggers.add_schedule(name="nightly", spec="daily 09:00", goal="first",
                          owner_user_id="U_OWNER")
    triggers.add_schedule(name="nightly", spec="daily 10:00", goal="second",
                          owner_user_id="U_OWNER")
    rows = [s for s in triggers.list_schedules() if s["name"] == "nightly"]
    assert len(rows) == 1 and rows[0]["goal"] == "second" and rows[0]["spec"] == "daily 10:00"


def test_cancelling_an_unknown_name_is_false(audit_env):
    assert triggers.cancel_schedule("no-such-schedule") is False
    assert triggers.cancel_schedule("") is False


def test_a_bad_spec_never_creates_a_row(audit_env):
    with pytest.raises(ValueError):
        triggers.add_schedule(name="broken", spec="every 0m", goal="x",
                              owner_user_id="U_OWNER")
    assert [s for s in triggers.list_schedules() if s["name"] == "broken"] == []


# ── firing ──

def test_a_due_schedule_queues_exactly_one_unattended_run(audit_env):
    sched = triggers.add_schedule(name="hourly-job", spec="every 1h", goal="do the thing",
                                  owner_user_id="U_OWNER", channel="C1")
    triggers._set_schedule_fields(sched["id"], next_run=time.time() - 1)
    ids = triggers.fire_due_schedules(now=time.time())
    assert len(ids) == 1
    run = runner.get_run(run_id_of(ids[0]))
    assert run["source"] == "schedule" and run["unattended"] in (1, True)
    assert run["goal"] == "do the thing"


def test_schedules_do_not_stack_but_next_run_still_advances(audit_env):
    """A slow run delays the next fire; it must not queue a backlog, and it must
    not leave the schedule permanently due."""
    sched = triggers.add_schedule(name="slow", spec="every 1h", goal="long job",
                                  owner_user_id="U_OWNER")
    triggers._set_schedule_fields(sched["id"], next_run=time.time() - 1)
    first = triggers.fire_due_schedules(now=time.time())
    assert len(first) == 1

    triggers._set_schedule_fields(sched["id"], next_run=time.time() - 1)
    second = triggers.fire_due_schedules(now=time.time())
    assert second == [], "a second run stacked on top of the first"
    row = [s for s in triggers.list_schedules() if s["name"] == "slow"][0]
    assert row["next_run"] > time.time(), "schedule left stuck in the due state"


def test_a_schedule_whose_run_is_rejected_still_advances(audit_env, monkeypatch):
    """Budget said no. The schedule must not retry every tick for the rest of
    the day."""
    monkeypatch.setenv("RUN_DAILY_LIMIT", "0")

    def rejected(**kwargs):
        raise runner.RunRejected("daily cap")

    monkeypatch.setattr(runner, "enqueue_run", rejected)
    sched = triggers.add_schedule(name="capped", spec="every 1h", goal="x",
                                  owner_user_id="U_OWNER")
    triggers._set_schedule_fields(sched["id"], next_run=time.time() - 1)
    assert triggers.fire_due_schedules(now=time.time()) == []
    row = [s for s in triggers.list_schedules() if s["name"] == "capped"][0]
    assert row["next_run"] > time.time()


def test_tick_expires_stale_approvals(audit_env):
    """The scheduler is what turns silence into a deny; without it a parked run
    waits forever."""
    import governor
    ap = governor.request_approval(1, "push_branch", {"branch": "x"})
    conn = governor._db()
    conn.execute("UPDATE approvals SET expires_at = ? WHERE id = ?",
                 (time.time() - 1, ap["id"]))
    conn.commit()
    conn.close()
    result = triggers.tick(now=time.time())
    assert ap["id"] in [a["id"] for a in result.get("expired", [])] or \
        governor.get_approval(ap["id"])["status"] == governor.EXPIRED
