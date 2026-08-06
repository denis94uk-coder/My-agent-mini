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
import re
import time
import threading
import logging

import agent
import critic
import governor
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
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL,
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
# Parked waiting on a human. Not claimable by a worker, but very much not
# finished — /runs shows it, and a decision puts it back in the queue.
AWAITING_APPROVAL = "awaiting_approval"

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


# Columns added after the first release. CREATE TABLE IF NOT EXISTS will not
# add them to a database that already has a `runs` table, so an install that
# has been running since before retries existed needs them patched in.
_MIGRATIONS = (
    ("attempts", "INTEGER NOT NULL DEFAULT 0"),
    ("next_attempt_at", "REAL"),
)


def _db():
    """Connection with both the core memory schema and the run tables present."""
    conn = memory.get_db()
    conn.executescript(_SCHEMA)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    for column, spec in _MIGRATIONS:
        if column not in existing:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {column} {spec}")
            logger.info(f"Migrated runs table: added {column}")
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


def max_retries() -> int:
    """How many times a *failed* run is retried before it stays failed."""
    return max(0, _int_env("RUN_MAX_RETRIES", 2))


def retry_backoff_seconds() -> int:
    return max(5, _int_env("RUN_RETRY_BACKOFF_SECONDS", 60))


def stuck_seconds() -> int:
    """
    How long a `running` run may go without a heartbeat before it's declared
    stuck. Generous by default: a single tool call can legitimately take
    minutes (a pytest quality gate, a slow clone).
    """
    return max(120, _int_env("RUN_STUCK_SECONDS", 1800))


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
    "attempts, next_attempt_at, label, worker, heartbeat, result, error, "
    "created, updated, started, finished"
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
        if event["kind"] == "compaction":
            # Everything before this point was folded into one note. Drop the
            # replayed history and continue from the fold — otherwise a resumed
            # run rebuilds the very context the fold existed to shrink.
            messages = [
                {"role": "user", "content": run["goal"]},
                {"role": "user", "content": event["content"]},
            ]
        elif event["kind"] == "assistant":
            messages.append({"role": "assistant", "content": event["content"]})
        elif event["kind"] == "tool_result":
            messages.append({
                "role": "user",
                "content": agent.tool_result_message(event["name"], event["content"]),
            })
        elif event["kind"] == "nudge":
            messages.append({"role": "user", "content": agent.NUDGE_MESSAGE})
        elif event["kind"] == "approval_requested":
            continue  # the decision's tool_result carries the outcome
        elif event["kind"] == "critic":
            # A rejected final answer: the agent's attempt, then the critique
            # it was sent back with. Both have to replay or a resumed run
            # would re-deliver the answer the critic already turned down.
            messages.append({"role": "user", "content": event["content"]})
    return messages


# ── Context compaction ──
#
# Durability created a problem it didn't solve: a resumed run replays its
# whole transcript, and a 25-step run with 4 KB tool results is ~25k tokens
# before the system prompt (operating manual + tool descriptions) is even
# added. On the free routes this repo targets, that overflows the context
# window and the run dies late, having done most of the work.
#
# So: past a threshold, fold the older turns into one progress note and keep
# only the recent ones verbatim. The fold is persisted, so a resume rebuilds
# the *compacted* history rather than the original — the saving survives a
# restart instead of being paid again every time.

COMPACTION_KEEP_RECENT = 6  # ~3 exchanges kept verbatim after a fold

# Minimum steps between folds. Without it, a run whose kept tail alone
# exceeds the limit would fold on every single step — each fold costing an AI
# call and a step, while freeing almost nothing.
COMPACTION_MIN_GAP_STEPS = 3


def context_limit_chars() -> int:
    """
    Fold when the working messages exceed this many characters.

    Chars, not tokens, because no provider here exposes a tokenizer. ~4 chars
    per token is the usual English approximation, so the 24000 default is
    roughly 6k tokens of transcript — leaving room for the system prompt on an
    8k-context model.
    """
    return max(2000, _int_env("RUN_CONTEXT_LIMIT_CHARS", 24000))


def _messages_size(messages: list[dict]) -> int:
    return sum(len(m.get("content") or "") for m in messages)


def _deterministic_digest(messages: list[dict]) -> str:
    """
    Fallback fold that needs no AI call.

    Used when the summarizer errors or the router is down. Cruder than a real
    summary, but it always works and it never costs a request — which matters
    when the paid route is the backup and the free one is what just failed.
    """
    lines = []
    for message in messages:
        content = (message.get("content") or "").strip().replace("\n", " ")
        if message.get("role") == "assistant":
            continue  # the model's own narration is the least useful part
        match = re.match(r"\[TOOL_RESULT for (\w+)\]\s*(.*)", content, re.DOTALL)
        if match:
            lines.append(f"- {match.group(1)} → {match.group(2)[:200]}")
        elif content.startswith("[CRITIC"):
            lines.append(f"- critic sent the answer back: {content[:200]}")
    if not lines:
        return "(earlier steps produced no tool results)"
    return "\n".join(lines[-25:])


def compact_messages(messages: list[dict], call_ai_fn=None) -> tuple[list[dict], str] | None:
    """
    Fold old turns into a progress note. Returns (new_messages, note) or None
    if there is nothing worth folding.

    The goal (message 0) and the last COMPACTION_KEEP_RECENT turns always
    survive verbatim: the goal because the run is judged against it, the tail
    because that's the work in flight.
    """
    if len(messages) <= COMPACTION_KEEP_RECENT + 1:
        return None

    goal = messages[0]
    older, recent = messages[1:-COMPACTION_KEEP_RECENT], messages[-COMPACTION_KEEP_RECENT:]
    if not older:
        return None

    digest = ""
    if call_ai_fn is not None:
        transcript = "\n".join(
            f"{m.get('role')}: {(m.get('content') or '')[:800]}" for m in older
        )
        try:
            digest = call_ai_fn(
                [{
                    "role": "user",
                    "content": (
                        "Summarize this agent's progress so far into a compact "
                        "handover note (max 250 words). Keep: what was actually done "
                        "and the real results, exact file paths, names, numbers, and "
                        "URLs, what failed and why, and what is still open. Drop: "
                        "narration, restatements, politeness. Terse note form.\n\n"
                        + transcript
                    ),
                }],
                "You write terse, accurate progress notes for an agent resuming work.",
            )
        except Exception as e:
            logger.warning(f"Compaction summary failed, using deterministic digest: {e}")
            digest = ""
        if isinstance(digest, str) and digest.startswith("❌"):
            digest = ""

    if not (digest or "").strip():
        digest = _deterministic_digest(older)

    note = (
        "[PROGRESS SO FAR — earlier steps of this same run, folded to save "
        "context. Treat as things you already did; do not repeat them]\n" + digest.strip()
    )
    return [goal, {"role": "user", "content": note}, *recent], note


def tool_steps(run_id: int) -> list[dict]:
    """The tool trail, as evidence for the critic gate."""
    return [
        {"tool": e["name"], "result": e["content"]}
        for e in get_events(run_id)
        if e["kind"] == "tool_result"
    ]


def critic_rounds_used(run_id: int) -> int:
    """How many times this run's final answer has already been sent back."""
    return sum(1 for e in get_events(run_id) if e["kind"] == "critic")


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
                "SELECT id FROM runs WHERE status = 'queued' "
                "AND (next_attempt_at IS NULL OR next_attempt_at <= ?) "
                "ORDER BY created, id LIMIT 1",
                (time.time(),),
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
    """
    Tools this run may not use at all — a flat no, with no way to ask.

    Since the governor, an unattended run has three levels rather than two:

      • blocked here      — never, not even with a human present. These spawn
                            more autonomous work; a run that can queue runs
                            can run away, and "ask a human" doesn't fix that.
      • approval required — EXTERNAL tier, gated by `_approval_gate`. The run
                            parks and asks instead of failing.
      • allowed           — everything else, plus EXTERNAL when the schedule
                            was explicitly marked `allow_risky`.

    With approvals switched off there is nobody to ask, so owner-only tools
    fall back to the original hard block rather than silently running.
    """
    if not run["unattended"]:
        return set()
    blocked = set(tools.UNATTENDED_BLOCKED_TOOLS)
    if not run["allow_risky"] and not governor.approvals_enabled():
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


# Failures worth retrying are the ones a later attempt could plausibly survive:
# a provider blip, a rate limit, a socket timeout. A misconfiguration will fail
# identically forever, so retrying it just burns quota against the same wall.
_PERMANENT_ERROR_MARKERS = (
    "no ai backend",
    "not authorized",
    "unknown tool",
)


def _is_retryable(error: str) -> bool:
    lowered = (error or "").lower()
    return not any(marker in lowered for marker in _PERMANENT_ERROR_MARKERS)


def _fail_or_retry(run: dict, error: str) -> bool:
    """
    Fail the run, or queue another attempt with backoff. True if it will retry.

    Distinct from `recover_interrupted_runs`, which handles the process dying
    mid-run. This handles the run itself erroring — the common cause on free
    routes being every provider cooling down at once.
    """
    run_id = run["id"]
    attempts = (run.get("attempts") or 0) + 1
    limit = max_retries()

    if attempts > limit or not _is_retryable(error):
        _update_run(run_id, attempts=attempts)
        _finish(run, "failed", error=error)
        return False

    delay = retry_backoff_seconds() * (2 ** (attempts - 1))
    _update_run(
        run_id, status="queued", attempts=attempts, worker="",
        next_attempt_at=time.time() + delay, error=error[:1000],
    )
    logger.warning(
        f"↻ Run #{run_id} failed ({error[:120]}) — retry {attempts}/{limit} in {delay}s"
    )
    return True


def _approval_gate(run: dict):
    """
    The predicate execute_step consults before running a tool.

    Only unattended runs are gated: if a human is in the conversation they are
    already the approval. Read and local-write tools go straight through —
    asking permission to run `ls` teaches people to approve without reading,
    which is how an approval queue stops being a safety mechanism.
    """
    if not run["unattended"] or not governor.approvals_enabled():
        return None

    blocked = _blocked_tools_for(run)

    def gate(tool_name: str, args: dict) -> bool:
        if tool_name in blocked:
            # A flat no. Let the block list answer it immediately rather than
            # parking the run on a request that could never be granted.
            return True
        if run["allow_risky"]:
            # The owner pre-authorised this schedule for external actions.
            return True
        return governor.tier_of(tool_name) != governor.EXTERNAL

    return gate


def _apply_pending_decision(run: dict, messages: list[dict]):
    """
    Carry out a decision made while the run was parked.

    Approved: the tool runs now, exactly as requested, and its result enters
    the transcript as if it had never paused. Denied or expired: a refusal
    goes in instead, and the agent has to finish without it.
    """
    decision = governor.undecided_action_for_run(run["id"])
    if not decision:
        return

    if decision["status"] == governor.APPROVED:
        args = dict(decision["args"])
        if decision["tool"] in tools.OWNER_ONLY_TOOLS:
            # The owner lock still applies; approval says *this* action is
            # sanctioned, not that the run has become the owner.
            args["_requesting_user_id"] = decision["decided_by"] or run["owner_user_id"]
        result = tools.run_tool(decision["tool"], args)
        logger.info(f"✅ Run #{run['id']} ran approved tool {decision['tool']}")
    else:
        result = governor.refusal_text(decision)

    add_event(run["id"], "tool_result", result, name=decision["tool"])
    messages.append({
        "role": "user",
        "content": agent.tool_result_message(decision["tool"], result),
    })
    governor.mark_executed(decision["id"])


def resume_after_decision(approval: dict) -> bool:
    """Put a decided run back in the queue. Called by the Slack commands."""
    run = get_run(approval["run_id"])
    if not run or run["status"] != AWAITING_APPROVAL:
        return False
    _update_run(run["id"], status="queued", next_attempt_at=None)
    logger.info(f"▶️ Run #{run['id']} re-queued after approval #{approval['id']}")
    return True


def execute_run(run: dict) -> dict:
    """
    Drive one run to completion (or to a budget limit), persisting every step.

    Returns the final run row. Safe to call on a run that already has events —
    it resumes from them rather than restarting the task.
    """
    if _CALL_AI is None:
        # Permanent by definition — retrying a missing backend just re-fails.
        _fail_or_retry(run, "No AI backend configured for the run engine.")
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
    _apply_pending_decision(run, messages)
    steps_used = run["steps_used"]
    critic_rounds = critic_rounds_used(run_id)
    # Large so the first fold can happen immediately; reset on every fold.
    steps_since_fold = COMPACTION_MIN_GAP_STEPS
    final_text = ""
    stop_reason = ""
    # The critique still outstanding if the round cap lands before the critic
    # is satisfied. It ships attached to the result rather than disappearing.
    unresolved = ""

    while True:
        # Between-step checks: a cancel or a budget limit lands here, which is
        # why steps are the unit of durability — never mid-tool.
        status = _run_status(run_id)
        if status == "cancelled":
            logger.info(f"🛑 Run #{run_id} cancelled")
            return get_run(run_id)
        if status not in ACTIVE_STATUSES:
            # Includes AWAITING_APPROVAL: the run parks and a decision
            # re-queues it, so a worker must let go of it here.
            return get_run(run_id)

        if steps_used >= run["max_steps"]:
            stop_reason = f"step budget ({run['max_steps']} steps)"
            break
        if time.time() >= deadline:
            stop_reason = f"time budget ({run['max_seconds']}s)"
            break

        if (
            _messages_size(messages) > context_limit_chars()
            and steps_since_fold >= COMPACTION_MIN_GAP_STEPS
        ):
            folded = compact_messages(messages, _CALL_AI)
            if folded:
                messages, note = folded
                steps_used += 1  # the summarizer is a real AI call
                steps_since_fold = 0
                add_event(run_id, "compaction", note)
                _update_run(run_id, steps_used=steps_used, heartbeat=time.time())
                logger.info(
                    f"🗜️ Run #{run_id} folded its context to "
                    f"{_messages_size(messages)} chars"
                )

        try:
            outcome = agent.execute_step(
                messages,
                _CALL_AI,
                full_prompt,
                user_id=run["owner_user_id"],
                conv_key=run["conv_key"] or f"run:{run_id}",
                allow_nudge=True,
                blocked_tools=blocked,
                approval_fn=_approval_gate(run),
            )
        except Exception as e:
            logger.warning(f"Run #{run_id} step failed: {e}", exc_info=True)
            add_event(run_id, "error", str(e)[:2000])
            if not _fail_or_retry(run, str(e)[:1000]):
                _notify(run, f"❌ Run #{run_id} failed: {str(e)[:300]}")
            # A retry keeps the transcript, so the next attempt resumes here
            # rather than redoing the work that already succeeded.
            return get_run(run_id)

        steps_used += 1
        steps_since_fold += 1
        _update_run(run_id, steps_used=steps_used, heartbeat=time.time())

        if outcome.kind == "paused":
            # Park durably. The assistant turn is persisted so the replayed
            # transcript still contains the request; the tool result arrives
            # only once a human decides.
            add_event(run_id, "assistant", outcome.response)
            approval = governor.request_approval(
                run_id, outcome.tool_name, outcome.tool_args,
                requested_for=run["owner_user_id"],
            )
            add_event(
                run_id, "approval_requested",
                f"{outcome.tool_name} (approval #{approval['id']})",
                name=outcome.tool_name,
            )
            _update_run(run_id, status=AWAITING_APPROVAL, worker="", heartbeat=time.time())
            _notify(run, governor.format_approval_request(approval, run))
            logger.info(f"⏸️ Run #{run_id} parked awaiting approval #{approval['id']}")
            return get_run(run_id)

        if outcome.kind == "tool":
            add_event(run_id, "assistant", outcome.response)
            add_event(run_id, "tool_result", outcome.tool_result, name=outcome.tool_name)
        elif outcome.kind == "nudge":
            add_event(run_id, "assistant", outcome.response)
            add_event(run_id, "nudge", agent.NUDGE_MESSAGE)
        else:
            final_text = agent.extract_final_text(outcome.response)

            # Critic gate: the agent decided it's done. Check that against
            # what the transcript actually shows before believing it.
            if critic.enabled() and critic_rounds < critic.max_rounds():
                verdict = critic.review(
                    run["goal"], tool_steps(run_id), final_text, _CALL_AI
                )
                steps_used += 1  # the critic is a real AI call; budget sees it
                _update_run(run_id, steps_used=steps_used, heartbeat=time.time())

                if not verdict.accepted:
                    critic_rounds += 1
                    revision = critic.revision_message(verdict.reason)
                    unresolved = verdict.reason
                    add_event(run_id, "assistant", outcome.response)
                    add_event(run_id, "critic", revision, name=f"round {critic_rounds}")
                    messages.append({"role": "assistant", "content": outcome.response})
                    messages.append({"role": "user", "content": revision})
                    logger.info(f"🔁 Run #{run_id} sent back by critic (round {critic_rounds})")
                    continue

                unresolved = ""
                add_event(run_id, "critic_ok", verdict.raw[:1000])

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

    if unresolved:
        final_text += critic.unresolved_note(unresolved)

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


def sweep_stuck_runs(now: float | None = None) -> list[int]:
    """
    Fail runs whose worker stopped heartbeating.

    The wall-clock budget is only checked *between* steps, so a run hung
    inside a tool call never reaches it — it stays `running` forever. That is
    not just an untidy row: `running` counts as active, so the schedule
    overlap guard would refuse to ever fire that schedule again.

    This does NOT re-queue the run. The worker thread may still be alive
    inside that tool, and Python cannot safely kill it; re-queueing would let
    a second worker execute the same steps concurrently. So the run is failed
    (with retries still available if the error is transient), the leaked
    worker is logged loudly, and the schedule behind it is unblocked.
    """
    now = time.time() if now is None else now
    cutoff = now - stuck_seconds()
    conn = _db()
    try:
        rows = conn.execute(
            f"SELECT {_RUN_COLUMNS} FROM runs WHERE status = 'running' "
            "AND heartbeat IS NOT NULL AND heartbeat < ?",
            (cutoff,),
        ).fetchall()
        stuck = [_row_to_run(r) for r in rows]
    finally:
        conn.close()

    for run in stuck:
        silent_for = int(now - (run["heartbeat"] or now))
        error = (
            f"No progress for {silent_for}s (worker '{run['worker']}' stopped "
            "heartbeating, most likely hung inside a tool call)."
        )
        logger.error(
            f"💀 Run #{run['id']} declared stuck after {silent_for}s — failing it. "
            f"Worker '{run['worker']}' may be leaked; restart the service to reclaim it."
        )
        add_event(run["id"], "error", error)
        if not _fail_or_retry(run, error):
            _notify(run, f"💀 Run #{run['id']} was stuck and has been stopped. {error}")
    return [r["id"] for r in stuck]


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
    icons = {"queued": "⏳", "running": "▶️", "done": "✅", "failed": "❌",
             "cancelled": "🛑", AWAITING_APPROVAL: "🖐️"}
    lines = []
    for r in runs:
        when = time.strftime("%m-%d %H:%M", time.localtime(r["created"]))
        detail = f"{r['steps_used']}/{r['max_steps']} steps"
        lines.append(
            f"{icons.get(r['status'], '•')} *#{r['id']}* [{r['source']}] {when} — "
            f"{r['goal'][:70]} _({r['status']}, {detail})_"
        )
    return "\n".join(lines)
