"""
Durable run engine — lets work outlive a single Slack turn.

The interactive path (`agent.run_agent_loop`) is synchronous: it lives inside
a Slack event handler, is capped at 10 tool steps, and keeps its whole state
in a local list. Restart the service mid-task and the work is gone.

A *run* is the durable version of that. Every step is written to SQLite
(`runs` + `run_events`) before the next one starts, so a run can be paused,
cancelled, inspected, and — after a crash or a `systemctl restart` — picked
up exactly where it stopped. Background worker threads pull queued runs, so
long work never blocks the chat.

Runs come from three places:
  • a human asking for background work (`start_background_run` tool)
  • a schedule firing (`triggers.py`) — this is what makes the agent
    self-starting rather than purely reactive
  • an unfinished plan being resumed (`triggers.py` plan sweeper)

Safety envelope (deliberately small — this is not the full policy layer):
  • per-run step and wall-clock budgets, enforced between steps
  • a daily cap on autonomously-created runs, so a bad schedule can't spin
  • unattended runs refuse OWNER_ONLY_TOOLS by default. The owner-lock in
    tools.py asks "is the human requesting this the owner?", which means
    nothing when a cron job is the caller and nobody is watching. A schedule
    must opt in explicitly (`allow_risky`) to deploy, push, or restart.
"""

import os
import time
import threading
import logging

import agent
import memory
import tools

logger = logging.getLogger("my-agent-mini")


# ── Schema ──

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',
    status TEXT NOT NULL DEFAULT 'queued',
    owner_user_id TEXT NOT NULL DEFAULT 'default',
    conv_key TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL DEFAULT '',
    thread_ts TEXT NOT NULL DEFAULT '',
    unattended INTEGER NOT NULL DEFAULT 1,
    allow_risky INTEGER NOT NULL DEFAULT 0,
    max_steps INTEGER NOT NULL DEFAULT 25,
    max_seconds INTEGER NOT NULL DEFAULT 900,
    steps_used INTEGER NOT NULL DEFAULT 0,
    resume_count INTEGER NOT NULL DEFAULT 0,
    label TEXT NOT NULL DEFAULT '',
    worker TEXT NOT NULL DEFAULT '',
    heartbeat REAL,
    result TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    created REAL NOT NULL,
    updated REAL NOT NULL,
    started REAL,
    finished REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_conv ON runs(conv_key);

CREATE TABLE IF NOT EXISTS run_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_run_events_run ON run_events(run_id, seq);
"""

# Statuses a run can be in. Only 'queued' is claimable by a worker.
ACTIVE_STATUSES = ("queued", "running")
TERMINAL_STATUSES = ("done", "failed", "cancelled")

# Sources that were not directly requested by a human right now. These are
# the ones the daily cap applies to.
AUTONOMOUS_SOURCES = ("schedule", "plan_resume")


def conv_target(conv_key: str) -> tuple[str, str]:
    """
    Split a conv_key back into the Slack channel + thread it came from.

    conv_key is built as f"{channel}:{thread_ts or 'main'}" (see
    bot.get_conv_key); this is the one place that knowledge is decoded.
    """
    channel, _, thread = (conv_key or "").partition(":")
    return channel, ("" if thread in ("", "main") else thread)


def _db():
    """Connection with both the core memory schema and the run tables present."""
    conn = memory.get_db()
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# ── Configuration ──

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def max_steps_default() -> int:
    return max(1, _int_env("RUN_MAX_STEPS", 25))


def max_seconds_default() -> int:
    return max(30, _int_env("RUN_MAX_SECONDS", 900))


def daily_limit() -> int:
    return max(0, _int_env("RUN_DAILY_LIMIT", 50))


def worker_count() -> int:
    return max(1, _int_env("RUN_WORKERS", 2))


def poll_seconds() -> float:
    return max(0.5, float(_int_env("RUN_POLL_SECONDS", 3)))


# A run needs an AI to call and (usually) somewhere to report back to.
# bot.py wires these up at startup; tests leave them unset and inject
# their own, which is why the engine never imports Slack itself.
_CALL_AI = None
_SYSTEM_PROMPT = "You are a helpful autonomous agent."
_POST_MESSAGE = None

_WORKERS_STARTED = False
_STOP = threading.Event()
_CLAIM_LOCK = threading.Lock()


def configure(call_ai_fn=None, system_prompt: str | None = None, post_message=None):
    """Wire the engine to an AI backend and a place to report results."""
    global _CALL_AI, _SYSTEM_PROMPT, _POST_MESSAGE
    if call_ai_fn is not None:
        _CALL_AI = call_ai_fn
    if system_prompt is not None:
        _SYSTEM_PROMPT = system_prompt
    if post_message is not None:
        _POST_MESSAGE = post_message


class RunRejected(Exception):
    """A run could not be queued (budget, missing config, bad input)."""


# ── Run lifecycle ──

_RUN_COLUMNS = (
    "id, goal, source, status, owner_user_id, conv_key, channel, thread_ts, "
    "unattended, allow_risky, max_steps, max_seconds, steps_used, resume_count, "
    "label, worker, heartbeat, result, error, created, updated, started, finished"
)


def _row_to_run(row) -> dict:
    return dict(zip(_RUN_COLUMNS.replace(" ", "").split(","), row))


def enqueue_run(
    goal: str,
    source: str = "manual",
    owner_user_id: str = "default",
    conv_key: str = "",
    channel: str = "",
    thread_ts: str = "",
    unattended: bool = True,
    allow_risky: bool = False,
    max_steps: int | None = None,
    max_seconds: int | None = None,
    label: str = "",
) -> int:
    """Queue a run and return its id. Raises RunRejected if a budget says no."""
    goal = (goal or "").strip()
    if not goal:
        raise RunRejected("A run needs a goal describing what to do.")

    now = time.time()
    conn = _db()
    try:
        if source in AUTONOMOUS_SOURCES:
            limit = daily_limit()
            started_today = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE source IN (?, ?) AND created > ?",
                (*AUTONOMOUS_SOURCES, now - 86400),
            ).fetchone()[0]
            if limit and started_today >= limit:
                raise RunRejected(
                    f"Daily autonomous run limit reached ({started_today}/{limit} in the "
                    "last 24h). Raise RUN_DAILY_LIMIT or wait — this cap exists so a "
                    "misfiring schedule can't spin forever."
                )

        cur = conn.execute(
            "INSERT INTO runs (goal, source, status, owner_user_id, conv_key, channel, "
            "thread_ts, unattended, allow_risky, max_steps, max_seconds, label, "
            "created, updated) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                goal, source, owner_user_id, conv_key, channel, thread_ts,
                1 if unattended else 0, 1 if allow_risky else 0,
                max_steps or max_steps_default(),
                max_seconds or max_seconds_default(),
                label, now, now,
            ),
        )
        conn.commit()
        run_id = cur.lastrowid
    finally:
        conn.close()

    logger.info(f"📥 Queued run #{run_id} ({source}): {goal[:80]}")
    return run_id


def get_run(run_id: int) -> dict | None:
    conn = _db()
    try:
        row = conn.execute(f"SELECT {_RUN_COLUMNS} FROM runs WHERE id = ?", (run_id,)).fetchone()
        return _row_to_run(row) if row else None
    finally:
        conn.close()


def list_runs(limit: int = 10, status: str = "") -> list[dict]:
    conn = _db()
    try:
        if status:
            rows = conn.execute(
                f"SELECT {_RUN_COLUMNS} FROM runs WHERE status = ? ORDER BY id DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_RUN_COLUMNS} FROM runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_run(r) for r in rows]
    finally:
        conn.close()


def cancel_run(run_id: int) -> bool:
    """
    Ask a run to stop. A queued run stops immediately; a running one stops at
    its next step boundary, since the engine re-reads status between steps.
    """
    conn = _db()
    try:
        cur = conn.execute(
            "UPDATE runs SET status = 'cancelled', updated = ?, finished = ? "
            "WHERE id = ? AND status IN (?, ?)",
            (time.time(), time.time(), run_id, *ACTIVE_STATUSES),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def add_event(run_id: int, kind: str, content: str, name: str = ""):
    """Append to a run's durable transcript. This is what makes resume possible."""
    conn = _db()
    try:
        seq = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM run_events WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO run_events (run_id, seq, kind, name, content, created) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, seq, kind, name, content[:8000], time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def get_events(run_id: int) -> list[dict]:
    conn = _db()
    try:
        rows = conn.execute(
            "SELECT seq, kind, name, content, created FROM run_events "
            "WHERE run_id = ? ORDER BY seq",
            (run_id,),
        ).fetchall()
        return [
            {"seq": r[0], "kind": r[1], "name": r[2], "content": r[3], "created": r[4]}
            for r in rows
        ]
    finally:
        conn.close()


def rebuild_messages(run: dict) -> list[dict]:
    """
    Reconstruct the working message list from the persisted transcript.

    This is the whole point of `run_events`: after a crash the engine replays
    the same assistant/tool turns the in-memory loop would have had, so the
    model resumes with its context intact instead of starting the task over.
    """
    messages = [{"role": "user", "content": run["goal"]}]
    for event in get_events(run["id"]):
        if event["kind"] == "assistant":
            messages.append({"role": "assistant", "content": event["content"]})
        elif event["kind"] == "tool_result":
            messages.append({
                "role": "user",
                "content": agent.tool_result_message(event["name"], event["content"]),
            })
        elif event["kind"] == "nudge":
            messages.append({"role": "user", "content": agent.NUDGE_MESSAGE})
    return messages


# ── Execution ──

def _notify(run: dict, text: str):
    """Report back to wherever the run came from (Slack, or just the log)."""
    if _POST_MESSAGE and run.get("channel"):
        try:
            _POST_MESSAGE(run["channel"], run.get("thread_ts") or "", text)
            return
        except Exception as e:
            logger.warning(f"Run #{run['id']} could not post to Slack: {e}")
    logger.info(f"[run #{run['id']}] {text[:500]}")


def _update_run(run_id: int, **fields):
    if not fields:
        return
    fields["updated"] = time.time()
    assignments = ", ".join(f"{k} = ?" for k in fields)
    conn = _db()
    try:
        conn.execute(f"UPDATE runs SET {assignments} WHERE id = ?", (*fields.values(), run_id))
        conn.commit()
    finally:
        conn.close()


def _claim_next_run(worker: str) -> dict | None:
    """
    Atomically take the oldest queued run.

    The lock plus the `status = 'queued'` guard in the UPDATE means two
    workers can never claim the same run, with or without RETURNING support.
    """
    with _CLAIM_LOCK:
        conn = _db()
        try:
            row = conn.execute(
                "SELECT id FROM runs WHERE status = 'queued' ORDER BY created, id LIMIT 1"
            ).fetchone()
            if not row:
                return None
            now = time.time()
            cur = conn.execute(
                "UPDATE runs SET status = 'running', worker = ?, heartbeat = ?, "
                "started = COALESCE(started, ?), updated = ? "
                "WHERE id = ? AND status = 'queued'",
                (worker, now, now, now, row[0]),
            )
            conn.commit()
            if cur.rowcount == 0:
                return None
        finally:
            conn.close()
    return get_run(row[0])


def _blocked_tools_for(run: dict) -> set:
    """Which tools this run may not use."""
    if not run["unattended"]:
        return set()
    # Never available unattended, opt-in or not: these spawn more autonomous
    # work, and a run that can queue runs can run away.
    blocked = set(tools.UNATTENDED_BLOCKED_TOOLS)
    if not run["allow_risky"]:
        blocked |= set(tools.OWNER_ONLY_TOOLS)
    return blocked


def _run_status(run_id: int) -> str:
    conn = _db()
    try:
        row = conn.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()
        return row[0] if row else "missing"
    finally:
        conn.close()


def _finish(run: dict, status: str, result: str = "", error: str = ""):
    _update_run(run["id"], status=status, result=result[:4000], error=error[:1000],
                finished=time.time())
    logger.info(f"🏁 Run #{run['id']} {status}")


def execute_run(run: dict) -> dict:
    """
    Drive one run to completion (or to a budget limit), persisting every step.

    Returns the final run row. Safe to call on a run that already has events —
    it resumes from them rather than restarting the task.
    """
    if _CALL_AI is None:
        _finish(run, "failed", error="No AI backend configured for the run engine.")
        return get_run(run["id"])

    run_id = run["id"]
    deadline = time.time() + run["max_seconds"]
    blocked = _blocked_tools_for(run)

    user_facts = memory.get_facts(run["owner_user_id"])
    full_prompt = agent.get_agent_system_prompt(_SYSTEM_PROMPT, user_facts)
    if run["unattended"]:
        full_prompt += (
            "\n\n═══════════════════════════════════════════════\n"
            "UNATTENDED RUN\n"
            "═══════════════════════════════════════════════\n"
            "Nobody is reading this as it happens — you were started by a schedule "
            "or resumed from an unfinished plan, not by someone typing right now. "
            "So: do not ask clarifying questions (state your assumption and "
            "continue), finish what you can, and end with a short report of what "
            "you actually did, what you found, and anything a human still needs "
            "to decide or run themselves.\n"
            + (
                "Tools that change state outside this server (deploys, pushes, "
                "service restarts) are blocked for this run — if the task needs "
                "one, do everything up to that point and say what's left.\n"
                if blocked else ""
            )
        )

    messages = rebuild_messages(run)
    steps_used = run["steps_used"]
    final_text = ""
    stop_reason = ""

    while True:
        # Between-step checks: a cancel or a budget limit lands here, which is
        # why steps are the unit of durability — never mid-tool.
        status = _run_status(run_id)
        if status == "cancelled":
            logger.info(f"🛑 Run #{run_id} cancelled")
            return get_run(run_id)
        if status not in ACTIVE_STATUSES:
            return get_run(run_id)

        if steps_used >= run["max_steps"]:
            stop_reason = f"step budget ({run['max_steps']} steps)"
            break
        if time.time() >= deadline:
            stop_reason = f"time budget ({run['max_seconds']}s)"
            break

        try:
            outcome = agent.execute_step(
                messages,
                _CALL_AI,
                full_prompt,
                user_id=run["owner_user_id"],
                conv_key=run["conv_key"] or f"run:{run_id}",
                allow_nudge=True,
                blocked_tools=blocked,
            )
        except Exception as e:
            logger.warning(f"Run #{run_id} step failed: {e}", exc_info=True)
            add_event(run_id, "error", str(e)[:2000])
            _finish(run, "failed", error=str(e)[:1000])
            _notify(run, f"❌ Run #{run_id} failed: {str(e)[:300]}")
            return get_run(run_id)

        steps_used += 1
        _update_run(run_id, steps_used=steps_used, heartbeat=time.time())

        if outcome.kind == "tool":
            add_event(run_id, "assistant", outcome.response)
            add_event(run_id, "tool_result", outcome.tool_result, name=outcome.tool_name)
        elif outcome.kind == "nudge":
            add_event(run_id, "assistant", outcome.response)
            add_event(run_id, "nudge", agent.NUDGE_MESSAGE)
        else:
            final_text = agent.extract_final_text(outcome.response)
            add_event(run_id, "final", final_text)
            break

    if stop_reason:
        # Out of budget: ask for the best answer available rather than
        # dropping the work on the floor with nothing to show.
        logger.warning(f"⚠️ Run #{run_id} hit {stop_reason}")
        try:
            final_text = agent.extract_final_text(
                _CALL_AI(
                    messages + [{
                        "role": "user",
                        "content": (
                            f"[SYSTEM] You have hit this run's {stop_reason}. Stop working "
                            "and report now: what you completed, what you found, and what "
                            "still needs doing. Do not call any more tools."
                        ),
                    }],
                    full_prompt,
                )
            )
        except Exception as e:
            final_text = f"(Run stopped at its {stop_reason}; no summary available: {e})"
        add_event(run_id, "final", final_text)
        final_text = f"{final_text}\n\n_(stopped at this run's {stop_reason})_"

    _finish(run, "done", result=final_text)
    _record_completion(run, final_text)
    _notify(run, _format_completion(run, final_text))
    return get_run(run_id)


def _format_completion(run: dict, final_text: str) -> str:
    origin = {
        "schedule": f"scheduled task `{run['label']}`" if run["label"] else "scheduled task",
        "plan_resume": "resumed plan",
        "manual": "background task",
    }.get(run["source"], run["source"])
    return f"🤖 *Finished {origin}* (run #{run['id']})\n\n{final_text[:3500]}"


def _record_completion(run: dict, final_text: str):
    """
    Write the run's outcome into conversation memory so the next Slack turn
    knows it happened. Without this, an autonomous run is invisible to the
    agent's own memory and it would happily redo the same work.
    """
    if not final_text:
        return
    conv_key = run["conv_key"] or f"run:{run['id']}"
    try:
        memory.add_message(
            conv_key, "assistant",
            f"[autonomous run #{run['id']} — {run['source']}] goal: {run['goal'][:300]}\n"
            f"result: {final_text[:1500]}",
        )
    except Exception as e:
        logger.debug(f"Could not record run #{run['id']} to memory: {e}")


# ── Workers ──

def recover_interrupted_runs() -> int:
    """
    Re-queue runs that were mid-flight when the process died.

    Their transcript is already on disk, so `rebuild_messages` puts the model
    back where it was. Runs that keep dying get failed rather than looping.
    """
    conn = _db()
    try:
        rows = conn.execute(
            "SELECT id, resume_count, max_steps FROM runs WHERE status = 'running'"
        ).fetchall()
        recovered = 0
        for run_id, resume_count, _ in rows:
            if resume_count >= 3:
                conn.execute(
                    "UPDATE runs SET status = 'failed', error = ?, finished = ?, updated = ? "
                    "WHERE id = ?",
                    ("Interrupted repeatedly (3 restarts) — giving up.",
                     time.time(), time.time(), run_id),
                )
                logger.warning(f"Run #{run_id} interrupted too many times, marking failed")
            else:
                conn.execute(
                    "UPDATE runs SET status = 'queued', resume_count = ?, worker = '', "
                    "updated = ? WHERE id = ?",
                    (resume_count + 1, time.time(), run_id),
                )
                recovered += 1
                logger.info(f"♻️ Re-queued interrupted run #{run_id} (resume {resume_count + 1})")
        conn.commit()
        return recovered
    finally:
        conn.close()


def _worker_loop(name: str):
    logger.info(f"👷 Run worker {name} started")
    while not _STOP.is_set():
        try:
            run = _claim_next_run(name)
            if run is None:
                _STOP.wait(poll_seconds())
                continue
            logger.info(f"▶️ Worker {name} executing run #{run['id']}: {run['goal'][:80]}")
            execute_run(run)
        except Exception:
            logger.exception(f"Run worker {name} crashed on a run; continuing")
            _STOP.wait(poll_seconds())
    logger.info(f"👷 Run worker {name} stopped")


def start_workers(count: int | None = None) -> int:
    """Start the background worker pool. Idempotent."""
    global _WORKERS_STARTED
    if _WORKERS_STARTED:
        return 0
    _WORKERS_STARTED = True
    _STOP.clear()

    recovered = recover_interrupted_runs()
    if recovered:
        logger.info(f"♻️ Recovered {recovered} interrupted run(s) from the last process")

    n = count or worker_count()
    for i in range(n):
        threading.Thread(target=_worker_loop, args=(f"w{i + 1}",), daemon=True).start()
    return n


def stop_workers():
    """Signal workers to stop after their current step. Used by tests."""
    global _WORKERS_STARTED
    _STOP.set()
    _WORKERS_STARTED = False


def format_runs(limit: int = 8) -> str:
    """Human-readable run list for Slack."""
    runs = list_runs(limit=limit)
    if not runs:
        return "No runs yet."
    icons = {"queued": "⏳", "running": "▶️", "done": "✅", "failed": "❌", "cancelled": "🛑"}
    lines = []
    for r in runs:
        when = time.strftime("%m-%d %H:%M", time.localtime(r["created"]))
        detail = f"{r['steps_used']}/{r['max_steps']} steps"
        lines.append(
            f"{icons.get(r['status'], '•')} *#{r['id']}* [{r['source']}] {when} — "
            f"{r['goal'][:70]} _({r['status']}, {detail})_"
        )
    return "\n".join(lines)
