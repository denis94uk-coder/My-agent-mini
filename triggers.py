"""
Triggers — the part that lets the agent start its own work.

Everything else in this bot begins with a human typing in Slack. This module
adds the two ways work can begin without that:

  1. **Schedules** — "every 30m", "daily 09:00", or a raw cron line. A ticker
     thread finds due schedules and queues a run.
  2. **The plan sweeper** — the `tasks` table already stores multi-step plans
     (`create_plan`), but nothing ever drove them forward: if the agent stopped
     after step 3 of 7, those steps sat there until a human said something.
     The sweeper finds plans that have gone quiet with steps still pending and
     queues a run to continue them.

The sweeper only fires when *both* the plan and its conversation have been
idle for `PLAN_STALE_SECONDS`. That idle check is what stops the agent from
talking over someone who is mid-conversation, or from resuming a plan that is
actually waiting on a human answer.

Times are interpreted in the server's local timezone.
"""

import os
import time
import logging
import threading

import governor
import memory
import runner

logger = logging.getLogger("my-agent-mini")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schedules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    spec TEXT NOT NULL,
    goal TEXT NOT NULL,
    owner_user_id TEXT NOT NULL DEFAULT 'default',
    channel TEXT NOT NULL DEFAULT '',
    thread_ts TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    allow_risky INTEGER NOT NULL DEFAULT 0,
    last_run REAL,
    next_run REAL NOT NULL,
    run_count INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_schedules_next ON schedules(next_run);
"""


def _db():
    # runner._db() brings the core memory tables *and* runs/run_events, which
    # the plan sweeper queries — so a first-ever tick on a fresh DB works.
    conn = runner._db()
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    return os.getenv(name, "true" if default else "false").strip().lower() in ("1", "true", "yes", "on")


# ── Schedule specs ──

_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_DOW_NAMES = {
    "sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6,
    "sunday": 0, "monday": 1, "tuesday": 2, "wednesday": 3,
    "thursday": 4, "friday": 5, "saturday": 6,
}

SPEC_HELP = (
    "Supported: `every 15m` / `every 2h` / `every 1d`, `hourly`, "
    "`daily 09:00`, `weekly mon 09:00`, or a raw 5-field cron line "
    "like `0 9 * * 1-5` (minute hour day-of-month month day-of-week, "
    "server local time)."
)


def parse_spec(spec: str) -> tuple[str, object]:
    """
    Normalize a schedule spec.

    Returns ("interval", seconds) or ("cron", "m h dom mon dow").
    Raises ValueError with SPEC_HELP for anything it can't read.
    """
    text = (spec or "").strip().lower()
    if not text:
        raise ValueError(f"Empty schedule. {SPEC_HELP}")

    parts = text.split()

    if parts[0] == "every":
        if len(parts) < 2:
            raise ValueError(f"'every' needs an interval, e.g. `every 15m`. {SPEC_HELP}")
        amount = "".join(parts[1:])
        # Accept "15m", "15 min", "15 minutes"
        digits = "".join(c for c in amount if c.isdigit())
        letters = "".join(c for c in amount if c.isalpha())
        if not digits or not letters:
            raise ValueError(f"Could not read the interval '{' '.join(parts[1:])}'. {SPEC_HELP}")
        unit = letters[0]
        if unit not in _UNIT_SECONDS:
            raise ValueError(f"Unknown time unit '{letters}'. {SPEC_HELP}")
        seconds = int(digits) * _UNIT_SECONDS[unit]
        if seconds < 60:
            raise ValueError("Minimum interval is 1 minute — anything faster just burns quota.")
        return "interval", seconds

    if parts[0] == "hourly":
        return "cron", "0 * * * *"

    if parts[0] == "daily":
        hour, minute = _parse_clock(parts[1] if len(parts) > 1 else "09:00")
        return "cron", f"{minute} {hour} * * *"

    if parts[0] == "weekly":
        if len(parts) < 2 or parts[1] not in _DOW_NAMES:
            raise ValueError(f"'weekly' needs a day, e.g. `weekly mon 09:00`. {SPEC_HELP}")
        dow = _DOW_NAMES[parts[1]]
        hour, minute = _parse_clock(parts[2] if len(parts) > 2 else "09:00")
        return "cron", f"{minute} {hour} * * {dow}"

    if parts[0] == "cron":
        parts = parts[1:]

    if len(parts) == 5:
        cron = " ".join(parts)
        _validate_cron(cron)
        return "cron", cron

    raise ValueError(f"Could not read schedule '{spec}'. {SPEC_HELP}")


def _parse_clock(text: str) -> tuple[int, int]:
    try:
        hour_str, _, minute_str = text.partition(":")
        hour, minute = int(hour_str), int(minute_str or 0)
    except ValueError:
        raise ValueError(f"Could not read the time '{text}' — use HH:MM, e.g. 09:00.")
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Time '{text}' is out of range — use HH:MM in 24h form.")
    return hour, minute


def _field_matches(field: str, value: int, wrap: int | None = None) -> bool:
    """Match one cron field (`*`, `*/n`, `a`, `a-b`, `a-b/n`, comma lists)."""
    for part in field.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if "/" in part:
            part, _, step_str = part.partition("/")
            step = int(step_str) if step_str.isdigit() and int(step_str) > 0 else 1
        if part in ("*", ""):
            low, high = 0, wrap if wrap is not None else 10**6
            if (value - low) % step == 0:
                return True
            continue
        if "-" in part:
            low_str, _, high_str = part.partition("-")
            if not (low_str.isdigit() and high_str.isdigit()):
                continue
            low, high = int(low_str), int(high_str)
            if low <= value <= high and (value - low) % step == 0:
                return True
            continue
        if part.isdigit():
            target = int(part)
            if wrap is not None and target == wrap:
                target = 0  # cron allows 7 for Sunday, 12-hour style wraps
            if target == value:
                return True
    return False


# minute, hour, day-of-month, month, day-of-week
_CRON_RANGES = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))
_CRON_FIELD_NAMES = ("minute", "hour", "day of month", "month", "day of week")


def _validate_cron_field(field: str, low: int, high: int, name: str):
    """Every number in one cron field has to be a number that field can hold."""
    for part in field.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"Empty {name} field. {SPEC_HELP}")
        value, _, step = part.partition("/")
        if step and (not step.isdigit() or int(step) == 0):
            raise ValueError(f"Bad step '/{step}' in the {name} field. {SPEC_HELP}")
        if value == "*":
            continue
        bounds = value.split("-")
        if len(bounds) > 2:
            raise ValueError(f"Bad range '{value}' in the {name} field. {SPEC_HELP}")
        for bound in bounds:
            if not bound.isdigit():
                raise ValueError(f"Bad {name} value '{bound}'. {SPEC_HELP}")
            if not (low <= int(bound) <= high):
                raise ValueError(
                    f"{name.capitalize()} {bound} is out of range ({low}-{high}). "
                    f"{SPEC_HELP}")
        if len(bounds) == 2 and int(bounds[0]) > int(bounds[1]):
            raise ValueError(f"Backwards {name} range '{value}'. {SPEC_HELP}")


def _validate_cron(cron: str):
    """
    Reject a cron line that cannot fire, not just one that cannot parse.

    Character-set validation alone accepted `0 25 * * *`: stored, listed as an
    active schedule, and silently never due, which is indistinguishable from a
    working schedule until someone notices the work never happened.
    """
    fields = cron.split()
    if len(fields) != 5:
        raise ValueError(f"A cron line needs 5 fields. {SPEC_HELP}")
    allowed = set("0123456789*/,- ")
    for field, (low, high), name in zip(fields, _CRON_RANGES, _CRON_FIELD_NAMES):
        if not field or set(field) - allowed:
            raise ValueError(f"Bad cron field '{field}'. {SPEC_HELP}")
        _validate_cron_field(field, low, high, name)


def _cron_matches(cron: str, when: float) -> bool:
    minute_f, hour_f, dom_f, month_f, dow_f = cron.split()
    t = time.localtime(when)
    cron_dow = (t.tm_wday + 1) % 7  # python: Mon=0 … Sun=6 → cron: Sun=0 … Sat=6
    return (
        _field_matches(minute_f, t.tm_min)
        and _field_matches(hour_f, t.tm_hour)
        and _field_matches(dom_f, t.tm_mday)
        and _field_matches(month_f, t.tm_mon)
        and _field_matches(dow_f, cron_dow, wrap=7)
    )


def next_run_after(spec: str, after: float | None = None) -> float:
    """Next fire time strictly after `after` (unix seconds)."""
    after = time.time() if after is None else after
    kind, value = parse_spec(spec)
    if kind == "interval":
        return after + value

    # Scan minute by minute from the next whole minute. Bounded at ~13 months
    # so an impossible date (Feb 30) fails loudly instead of hanging.
    #
    # Cron is matched in the server's local time, so on a DST fall-back the
    # same wall-clock minute happens twice, an hour apart: `daily 01:30` used
    # to fire at 01:30 BST and again at 01:30 GMT. A schedule that opens a PR
    # or deploys would do it twice. Skipping candidates that land on the same
    # local Y-M-D H:M as the run we are scheduling after collapses the repeat
    # back to one firing. (The spring-forward hour simply does not exist; that
    # schedule moves to the next day, which is the safe direction.)
    fired_minute = time.localtime(after)[:5]
    candidate = (int(after) // 60) * 60 + 60
    for _ in range(60 * 24 * 400):
        if _cron_matches(value, candidate) and time.localtime(candidate)[:5] != fired_minute:
            return float(candidate)
        candidate += 60
    raise ValueError(f"Schedule '{spec}' never matches a real date.")


# ── Schedule CRUD ──

_COLUMNS = (
    "id, name, spec, goal, owner_user_id, channel, thread_ts, enabled, "
    "allow_risky, last_run, next_run, run_count, created"
)


def _row_to_schedule(row) -> dict:
    return dict(zip(_COLUMNS.replace(" ", "").split(","), row))


def add_schedule(
    name: str,
    spec: str,
    goal: str,
    owner_user_id: str = "default",
    channel: str = "",
    thread_ts: str = "",
    allow_risky: bool = False,
) -> dict:
    """Create or replace a named schedule. Raises ValueError on a bad spec."""
    name = (name or "").strip()
    goal = (goal or "").strip()
    if not name:
        raise ValueError("A schedule needs a short name so you can cancel it later.")
    if not goal:
        raise ValueError("A schedule needs a goal describing what to do when it fires.")

    next_run = next_run_after(spec)  # validates the spec too
    now = time.time()
    conn = _db()
    try:
        conn.execute(
            "INSERT INTO schedules (name, spec, goal, owner_user_id, channel, thread_ts, "
            "enabled, allow_risky, next_run, created) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET spec = excluded.spec, goal = excluded.goal, "
            "channel = excluded.channel, thread_ts = excluded.thread_ts, "
            "allow_risky = excluded.allow_risky, next_run = excluded.next_run, enabled = 1",
            (name, spec.strip(), goal, owner_user_id, channel, thread_ts,
             1 if allow_risky else 0, next_run, now),
        )
        conn.commit()
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM schedules WHERE name = ?", (name,)
        ).fetchone()
    finally:
        conn.close()
    logger.info(f"⏰ Schedule '{name}' set ({spec}) — next run {time.ctime(next_run)}")
    return _row_to_schedule(row)


def list_schedules(include_disabled: bool = True) -> list[dict]:
    conn = _db()
    try:
        sql = f"SELECT {_COLUMNS} FROM schedules"
        if not include_disabled:
            sql += " WHERE enabled = 1"
        return [_row_to_schedule(r) for r in conn.execute(sql + " ORDER BY next_run").fetchall()]
    finally:
        conn.close()


def cancel_schedule(name_or_id: str) -> bool:
    conn = _db()
    try:
        cur = conn.execute(
            "DELETE FROM schedules WHERE name = ? OR CAST(id AS TEXT) = ?",
            (str(name_or_id).strip(), str(name_or_id).strip()),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def fire_due_schedules(now: float | None = None) -> list[int]:
    """Queue a run for every schedule that is due. Returns the new run ids."""
    now = time.time() if now is None else now
    conn = _db()
    try:
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM schedules WHERE enabled = 1 AND next_run <= ?", (now,)
        ).fetchall()
    finally:
        conn.close()

    run_ids = []
    for row in rows:
        sched = _row_to_schedule(row)

        # Overlap guard: a schedule firing faster than its runs finish would
        # otherwise stack them up — on a 1 GB box that is how you lose the
        # process. The plan sweeper has always had this check; schedules
        # needed it too. next_run still advances below, so a slow run delays
        # the next fire instead of queueing a backlog to run all at once.
        if not _bool_env("SCHEDULE_ALLOW_OVERLAP", False) and _schedule_run_active(sched["name"]):
            logger.info(
                f"⏭️ Schedule '{sched['name']}' skipped: its previous run is still going"
            )
            try:
                _set_schedule_fields(
                    sched["id"], last_run=now, next_run=next_run_after(sched["spec"], now)
                )
            except ValueError:
                _set_schedule_fields(sched["id"], enabled=0)
            continue

        try:
            run_id = runner.enqueue_run(
                goal=sched["goal"],
                source="schedule",
                owner_user_id=sched["owner_user_id"],
                channel=sched["channel"],
                thread_ts=sched["thread_ts"],
                unattended=True,
                allow_risky=bool(sched["allow_risky"]),
                label=sched["name"],
            )
            run_ids.append(run_id)
        except runner.RunRejected as e:
            # Budget said no. Still advance next_run, or this schedule would
            # retry every tick for the rest of the day.
            logger.warning(f"Schedule '{sched['name']}' not queued: {e}")
        except Exception:
            logger.exception(f"Schedule '{sched['name']}' failed to queue")

        try:
            upcoming = next_run_after(sched["spec"], now)
        except ValueError:
            logger.error(f"Schedule '{sched['name']}' has an unusable spec; disabling it")
            _set_schedule_fields(sched["id"], enabled=0)
            continue
        _set_schedule_fields(
            sched["id"], last_run=now, next_run=upcoming, run_count=sched["run_count"] + 1
        )
    return run_ids


def _schedule_run_active(name: str) -> bool:
    """Is a run from this schedule still going?"""
    conn = _db()
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM runs WHERE source = 'schedule' AND label = ? "
            "AND status IN (?, ?)",
            (name, *runner.ACTIVE_STATUSES),
        ).fetchone()[0] > 0
    finally:
        conn.close()


def _set_schedule_fields(schedule_id: int, **fields):
    assignments = ", ".join(f"{k} = ?" for k in fields)
    conn = _db()
    try:
        conn.execute(
            f"UPDATE schedules SET {assignments} WHERE id = ?", (*fields.values(), schedule_id)
        )
        conn.commit()
    finally:
        conn.close()


def format_schedules(show_goals: bool = True) -> str:
    """
    Human-readable schedule list for Slack.

    `show_goals=False` hides the goal text for the same reason format_runs
    does: the list is world-readable, the goals are the owner's.
    """
    scheds = list_schedules()
    if not scheds:
        return "No schedules set. Ask me to schedule something, e.g. _\"every weekday at 9am, check the repo for open PRs and summarize them\"_."
    lines = []
    for s in scheds:
        state = "" if s["enabled"] else " _(disabled)_"
        nxt = time.strftime("%a %m-%d %H:%M", time.localtime(s["next_run"]))
        risky = " ⚠️ risky-tools-allowed" if s["allow_risky"] else ""
        goal = s["goal"][:120] if show_goals else "goal hidden — owner only"
        lines.append(
            f"⏰ *{s['name']}* — `{s['spec']}` → next {nxt}{state}{risky}\n"
            f"    _{goal}_ ({s['run_count']} run(s) so far)"
        )
    return "\n".join(lines)


# ── Plan sweeper ──

def stale_plan_conv_keys(now: float | None = None, stale_seconds: int | None = None) -> list[str]:
    """
    Conversations with unfinished plans that have gone quiet.

    Quiet means both the plan and the conversation itself have been idle —
    an actively chatting human is never interrupted by a resume.

    A plan with a step marked `blocked` is not quiet, it is waiting: resuming
    it just spends AI calls rediscovering that it still needs a human answer.
    Mark the step `blocked` (the `update_task` tool already offers that status)
    and the sweeper leaves the plan alone until someone unblocks it.
    """
    now = time.time() if now is None else now
    stale = stale_seconds if stale_seconds is not None else _int_env("PLAN_STALE_SECONDS", 600)
    cutoff = now - stale

    conn = _db()
    try:
        rows = conn.execute(
            "SELECT conv_key, MAX(updated) FROM tasks WHERE status NOT IN ('done', 'blocked') "
            "AND conv_key NOT IN (SELECT conv_key FROM tasks WHERE status = 'blocked') "
            "GROUP BY conv_key HAVING MAX(updated) < ?",
            (cutoff,),
        ).fetchall()
        quiet = []
        for conv_key, _ in rows:
            last_msg = conn.execute(
                "SELECT MAX(timestamp) FROM conversations WHERE conv_key = ?", (conv_key,)
            ).fetchone()[0]
            if last_msg is not None and last_msg >= cutoff:
                continue  # someone is still talking in this thread
            quiet.append(conv_key)
        return quiet
    finally:
        conn.close()


def _resume_count(conv_key: str, since: float) -> int:
    conn = _db()
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM runs WHERE conv_key = ? AND source = 'plan_resume' "
            "AND created > ?",
            (conv_key, since),
        ).fetchone()[0]
    finally:
        conn.close()


def _has_active_run(conv_key: str) -> bool:
    conn = _db()
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM runs WHERE conv_key = ? AND status IN (?, ?)",
            (conv_key, *runner.ACTIVE_STATUSES),
        ).fetchone()[0] > 0
    finally:
        conn.close()


def _plan_owner(conv_key: str) -> str:
    conn = _db()
    try:
        row = conn.execute(
            "SELECT user_id FROM tasks WHERE conv_key = ? LIMIT 1", (conv_key,)
        ).fetchone()
        return row[0] if row else "default"
    finally:
        conn.close()


def _resume_goal(conv_key: str) -> str:
    plan = memory.get_plan(conv_key)
    steps = "\n".join(
        f"  {p['step_no']}. [{p['status']}] {p['description']}" for p in plan
    )
    pending = [p for p in plan if p["status"] != "done"]
    return (
        "Continue an unfinished plan from an earlier conversation. Nobody nudged "
        "you — this run exists because the plan still has open steps and the "
        "thread went quiet.\n\n"
        f"THE PLAN so far:\n{steps}\n\n"
        f"{len(pending)} step(s) are still open. Use list_tasks to re-check state, "
        "do the open steps, and call update_task as you finish each one. Use "
        "memory_search to recall the context of this work before acting. If a step "
        "genuinely cannot be done without a human decision, mark what you did, stop, "
        "and say clearly what you need."
    )


def sweep_stale_plans(now: float | None = None) -> list[int]:
    """Queue continuation runs for quiet, unfinished plans. Returns run ids."""
    if not _bool_env("PLAN_AUTO_RESUME", True):
        return []

    now = time.time() if now is None else now
    max_resumes = _int_env("PLAN_MAX_RESUMES", 3)
    queued = []

    for conv_key in stale_plan_conv_keys(now):
        if _has_active_run(conv_key):
            continue
        if max_resumes and _resume_count(conv_key, now - 86400) >= max_resumes:
            logger.debug(f"Plan {conv_key} hit its resume cap; leaving it alone")
            continue

        channel, thread_ts = runner.conv_target(conv_key)
        try:
            run_id = runner.enqueue_run(
                goal=_resume_goal(conv_key),
                source="plan_resume",
                owner_user_id=_plan_owner(conv_key),
                conv_key=conv_key,
                channel=channel,
                thread_ts=thread_ts,
                unattended=True,
                label=f"plan:{conv_key}",
            )
            queued.append(run_id)
            logger.info(f"📋 Queued plan continuation for {conv_key} as run #{run_id}")
        except runner.RunRejected as e:
            logger.warning(f"Plan resume for {conv_key} not queued: {e}")
        except Exception:
            logger.exception(f"Plan resume for {conv_key} failed to queue")
    return queued


# ── Ticker ──

_STOP = threading.Event()
_STARTED = False


def tick(now: float | None = None) -> dict:
    """One scheduler pass. Exposed separately so tests don't need the thread."""
    now = time.time() if now is None else now
    fired = []
    swept = []
    expired = []
    stuck = []
    try:
        fired = fire_due_schedules(now)
    except Exception:
        logger.exception("Schedule tick failed")
    try:
        swept = sweep_stale_plans(now)
    except Exception:
        logger.exception("Plan sweep failed")
    try:
        # Silence is a deny. Expired requests unpark their run so it can
        # report that it never got an answer instead of sitting forever.
        for approval in governor.expire_stale_approvals(now):
            expired.append(approval["id"])
            runner.resume_after_decision(approval)
    except Exception:
        logger.exception("Approval expiry sweep failed")
    try:
        stuck = runner.sweep_stuck_runs(now)
    except Exception:
        logger.exception("Stuck-run sweep failed")
    return {"schedules_fired": fired, "plans_resumed": swept,
            "approvals_expired": expired, "runs_unstuck": stuck}


def _ticker_loop():
    interval = max(10, _int_env("SCHEDULER_TICK_SECONDS", 30))
    logger.info(f"⏰ Scheduler started (tick every {interval}s)")
    while not _STOP.is_set():
        result = tick()
        if any(result.values()):
            logger.info(
                f"⏰ Tick queued {len(result['schedules_fired'])} scheduled run(s), "
                f"{len(result['plans_resumed'])} plan continuation(s)"
            )
        _STOP.wait(interval)
    logger.info("⏰ Scheduler stopped")


def start_scheduler() -> bool:
    """Start the ticker thread. Idempotent; honours SCHEDULER_ENABLED."""
    global _STARTED
    if _STARTED:
        return False
    if not _bool_env("SCHEDULER_ENABLED", True):
        logger.info("⏰ Scheduler disabled (SCHEDULER_ENABLED=false)")
        return False
    _STARTED = True
    _STOP.clear()
    threading.Thread(target=_ticker_loop, daemon=True).start()
    return True


def stop_scheduler():
    global _STARTED
    _STOP.set()
    _STARTED = False
