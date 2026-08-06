"""
Governor — risk tiers, human approval, and cost accounting for autonomous work.

Phases 1-4 gave the bot the ability to act on its own. The safety envelope
that shipped with them is binary: an unattended run either may use a tool or
may not, decided before the run starts. That is too blunt in both directions.
It blocks a scheduled task that legitimately needs to open one PR, and a
schedule with `allow_risky` set can deploy at 3am with nobody consulted.

This adds the middle option — *ask* — plus the accounting that tells you what
the autonomy is costing.

Three parts:

  1. **Risk tiers.** Every tool is classified READ / WRITE_LOCAL / EXTERNAL.
     Unclassified tools are treated as EXTERNAL, because the failure mode of
     forgetting to classify a new deploy tool is far worse than the friction
     of an unnecessary approval. A test asserts every registered tool is
     explicitly classified, so that default never actually fires in practice.

  2. **Approval queue.** An unattended run that reaches an EXTERNAL tool does
     not fail and does not proceed: it *pauses durably*, records what it wants
     to do, and asks in Slack. `/approve <id>` resumes it and the tool runs;
     `/deny <id>` resumes it with a refusal it must work around. Unanswered
     requests expire, which is a deny — silence must never become consent.

  3. **Accounting.** AI calls are counted per run and per provider per day,
     with paid routes tracked separately and capped. The free route is
     best-effort; the paid one is a real monthly budget, and an agent that can
     start its own work can also spend that budget while nobody is looking.
"""

import os
import json
import time
import logging

import memory

logger = logging.getLogger("my-agent-mini")


# ── Risk tiers ──

READ = "read"                 # observes only: search, fetch, read files, recall
WRITE_LOCAL = "write_local"   # changes this server: shell, files, workspace
EXTERNAL = "external"         # changes the world outside it, hard to undo

TIER_ORDER = (READ, WRITE_LOCAL, EXTERNAL)

# Classification is explicit, per tool, and reviewed when tools are added.
# See `test_every_registered_tool_has_a_tier` — adding a tool without a tier
# fails the suite rather than silently inheriting a default.
TOOL_TIERS = {
    # READ
    "web_search": READ,
    "fetch_url": READ,
    "get_weather": READ,
    "memory_search": READ,
    "graph_recall": READ,
    "graph_inspect": READ,
    "list_files": READ,
    "read_file": READ,
    "list_tasks": READ,
    "github_read_file": READ,
    "github_list_issues": READ,
    "repo_read_file": READ,
    "repo_list_files": READ,
    "server_health": READ,
    "list_schedules": READ,
    "run_status": READ,
    # WRITE_LOCAL — reversible, confined to this box
    "run_shell": WRITE_LOCAL,
    "run_python": WRITE_LOCAL,
    "write_file": WRITE_LOCAL,
    "remember": WRITE_LOCAL,
    "create_plan": WRITE_LOCAL,
    "update_task": WRITE_LOCAL,
    "clone_repo": WRITE_LOCAL,
    "repo_write_file": WRITE_LOCAL,
    "repo_edit_file": WRITE_LOCAL,
    "repo_check": WRITE_LOCAL,
    "scaffold_site": WRITE_LOCAL,
    # EXTERNAL — visible outside this server, or commits it to future action
    "github_write_file": EXTERNAL,
    "github_create_issue": EXTERNAL,
    "push_branch": EXTERNAL,
    "deploy_static_site": EXTERNAL,
    "restart_service": EXTERNAL,
    "schedule_task": EXTERNAL,
    "cancel_schedule": EXTERNAL,
    "start_background_run": EXTERNAL,
}


def tier_of(tool_name: str) -> str:
    """
    Risk tier for a tool. Unknown tools are EXTERNAL — fail safe, not silent.

    A new tool that nobody classified is exactly the case where guessing wrong
    is expensive, so it gets the treatment that asks a human first.
    """
    return TOOL_TIERS.get(tool_name, EXTERNAL)


def _bool_env(name: str, default: bool) -> bool:
    return os.getenv(name, "true" if default else "false").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def approvals_enabled() -> bool:
    return _bool_env("APPROVALS_ENABLED", True)


def approval_timeout_seconds() -> int:
    """Unanswered approvals expire after this long. Expiry is a deny."""
    return max(60, _int_env("APPROVAL_TIMEOUT_SECONDS", 86400))


# ── Schema ──

_SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    tool TEXT NOT NULL,
    args_json TEXT NOT NULL DEFAULT '{}',
    tier TEXT NOT NULL DEFAULT 'external',
    status TEXT NOT NULL DEFAULT 'pending',
    requested_for TEXT NOT NULL DEFAULT '',
    decided_by TEXT NOT NULL DEFAULT '',
    decided_at REAL,
    executed INTEGER NOT NULL DEFAULT 0,
    note TEXT NOT NULL DEFAULT '',
    created REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals(run_id, status);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status);

CREATE TABLE IF NOT EXISTS provider_usage (
    provider TEXT NOT NULL,
    day TEXT NOT NULL,
    paid INTEGER NOT NULL DEFAULT 0,
    calls INTEGER NOT NULL DEFAULT 0,
    input_chars INTEGER NOT NULL DEFAULT 0,
    output_chars INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, day)
);
"""

PENDING = "pending"
APPROVED = "approved"
DENIED = "denied"
EXPIRED = "expired"


def _db():
    conn = memory.get_db()
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


_COLUMNS = (
    "id, run_id, tool, args_json, tier, status, requested_for, decided_by, "
    "decided_at, executed, note, created, expires_at"
)


def _row(row) -> dict:
    data = dict(zip(_COLUMNS.replace(" ", "").split(","), row))
    try:
        data["args"] = json.loads(data["args_json"])
    except (json.JSONDecodeError, TypeError):
        data["args"] = {}
    return data


# ── Approval queue ──

def request_approval(run_id: int, tool: str, args: dict, requested_for: str = "") -> dict:
    """Record that a run wants to run `tool` and is waiting on a human."""
    safe_args = {k: v for k, v in (args or {}).items() if not k.startswith("_")}
    now = time.time()
    conn = _db()
    try:
        cur = conn.execute(
            "INSERT INTO approvals (run_id, tool, args_json, tier, status, "
            "requested_for, created, expires_at) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)",
            (run_id, tool, json.dumps(safe_args, default=str)[:4000], tier_of(tool),
             requested_for, now, now + approval_timeout_seconds()),
        )
        conn.commit()
        approval_id = cur.lastrowid
    finally:
        conn.close()
    logger.info(f"🖐️ Run #{run_id} is waiting for approval #{approval_id} to run {tool}")
    return get_approval(approval_id)


def get_approval(approval_id: int) -> dict | None:
    conn = _db()
    try:
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM approvals WHERE id = ?", (approval_id,)
        ).fetchone()
        return _row(row) if row else None
    finally:
        conn.close()


def list_approvals(status: str = PENDING, limit: int = 20) -> list[dict]:
    conn = _db()
    try:
        if status:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM approvals WHERE status = ? ORDER BY id DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM approvals ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row(r) for r in rows]
    finally:
        conn.close()


def pending_for_run(run_id: int) -> dict | None:
    conn = _db()
    try:
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM approvals WHERE run_id = ? AND status = 'pending' "
            "ORDER BY id LIMIT 1",
            (run_id,),
        ).fetchone()
        return _row(row) if row else None
    finally:
        conn.close()


def undecided_action_for_run(run_id: int) -> dict | None:
    """
    A decision that has been made but not yet carried out.

    This is what a resuming run looks for: the human said yes or no while the
    run was parked, and the tool still has to be executed (or refused) before
    the loop continues.
    """
    conn = _db()
    try:
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM approvals WHERE run_id = ? AND executed = 0 "
            "AND status IN ('approved', 'denied', 'expired') ORDER BY id LIMIT 1",
            (run_id,),
        ).fetchone()
        return _row(row) if row else None
    finally:
        conn.close()


def decide(approval_id: int, approved: bool, decided_by: str = "", note: str = "") -> dict | None:
    """Approve or deny a pending request. Returns the updated row, or None."""
    conn = _db()
    try:
        cur = conn.execute(
            "UPDATE approvals SET status = ?, decided_by = ?, decided_at = ?, note = ? "
            "WHERE id = ? AND status = 'pending'",
            (APPROVED if approved else DENIED, decided_by, time.time(), note[:500], approval_id),
        )
        conn.commit()
        if cur.rowcount == 0:
            return None
    finally:
        conn.close()
    logger.info(f"🖐️ Approval #{approval_id} {'approved' if approved else 'denied'} by {decided_by or 'unknown'}")
    return get_approval(approval_id)


def mark_executed(approval_id: int):
    conn = _db()
    try:
        conn.execute("UPDATE approvals SET executed = 1 WHERE id = ?", (approval_id,))
        conn.commit()
    finally:
        conn.close()


def expire_stale_approvals(now: float | None = None) -> list[dict]:
    """
    Time out unanswered requests.

    Silence is a deny, not a yes: an approval nobody looked at must not become
    permission just because time passed. The run is unparked so it can report
    that it never got an answer, rather than sitting parked forever.
    """
    now = time.time() if now is None else now
    conn = _db()
    try:
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM approvals WHERE status = 'pending' AND expires_at <= ?",
            (now,),
        ).fetchall()
        expired = [_row(r) for r in rows]
        for item in expired:
            conn.execute(
                "UPDATE approvals SET status = 'expired', decided_at = ? WHERE id = ?",
                (now, item["id"]),
            )
        conn.commit()
    finally:
        conn.close()
    for item in expired:
        logger.info(f"⌛ Approval #{item['id']} for {item['tool']} expired unanswered")
    return expired


def refusal_text(approval: dict) -> str:
    """What the agent sees in place of the tool result when it doesn't get a yes."""
    if approval["status"] == EXPIRED:
        return (
            f"❌ Not approved: nobody answered the approval request for "
            f"'{approval['tool']}' in time, so it was not run. Treat this as a no. "
            "Finish the rest of the work and report clearly that this step still "
            "needs a human."
        )
    note = f" Reason given: {approval['note']}" if approval.get("note") else ""
    return (
        f"❌ Denied: a human declined the request to run '{approval['tool']}'.{note} "
        "Do not try to achieve the same effect another way. Finish the rest of the "
        "work and report what was declined."
    )


def format_approval_request(approval: dict, run: dict) -> str:
    """The Slack message asking for a decision."""
    args = approval.get("args") or {}
    arg_lines = "\n".join(f"    • {k}: {str(v)[:200]}" for k, v in args.items()) or "    (none)"
    hours = round(approval_timeout_seconds() / 3600, 1)
    return (
        f"🖐️ *Approval needed* — run #{run['id']} is paused\n\n"
        f"*Wants to run:* `{approval['tool']}` _({approval['tier']})_\n"
        f"*With:*\n{arg_lines}\n"
        f"*For the goal:* {run['goal'][:300]}\n\n"
        f"`/approve {approval['id']}` or `/deny {approval['id']} <optional reason>`\n"
        f"_Expires in {hours}h; no answer counts as a deny._"
    )


def format_approvals() -> str:
    pending = list_approvals(PENDING)
    if not pending:
        return "No approvals waiting."
    lines = []
    for item in pending:
        left = max(0, int((item["expires_at"] - time.time()) / 60))
        lines.append(
            f"🖐️ *#{item['id']}* run #{item['run_id']} — `{item['tool']}` "
            f"_({left} min left)_"
        )
    return "\n".join(lines) + "\n\n_`/approve <id>` or `/deny <id> <reason>`_"


# ── Cost accounting ──

def paid_daily_limit() -> int:
    """Max calls to paid routes per day. 0 disables the cap."""
    return max(0, _int_env("PAID_DAILY_LIMIT", 200))


def paid_provider_names() -> set:
    """
    Which routes cost real money.

    The free routes are best-effort and rate-limited; the paid backup is a
    real monthly budget. An agent that starts its own work can also spend that
    budget unattended, so it gets counted and capped separately.
    """
    raw = os.getenv("PAID_PROVIDERS", "merge")
    return {name.strip().lower() for name in raw.split(",") if name.strip()}


def is_paid(provider_name: str) -> bool:
    lowered = (provider_name or "").lower()
    return any(paid in lowered for paid in paid_provider_names())


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def record_ai_call(provider_name: str, input_chars: int = 0, output_chars: int = 0):
    """Count one AI call. Never raises — accounting must not break a reply."""
    try:
        conn = _db()
        try:
            conn.execute(
                "INSERT INTO provider_usage (provider, day, paid, calls, input_chars, "
                "output_chars) VALUES (?, ?, ?, 1, ?, ?) "
                "ON CONFLICT(provider, day) DO UPDATE SET calls = calls + 1, "
                "input_chars = input_chars + excluded.input_chars, "
                "output_chars = output_chars + excluded.output_chars",
                (provider_name, _today(), 1 if is_paid(provider_name) else 0,
                 input_chars, output_chars),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.debug(f"Usage accounting skipped: {e}")


def paid_calls_today() -> int:
    conn = _db()
    try:
        return conn.execute(
            "SELECT COALESCE(SUM(calls), 0) FROM provider_usage WHERE day = ? AND paid = 1",
            (_today(),),
        ).fetchone()[0]
    finally:
        conn.close()


def paid_budget_exhausted() -> bool:
    """True when paid routes should be skipped for the rest of the day."""
    limit = paid_daily_limit()
    return bool(limit) and paid_calls_today() >= limit


def usage_summary(days: int = 7) -> list[dict]:
    conn = _db()
    try:
        rows = conn.execute(
            "SELECT provider, paid, SUM(calls), SUM(input_chars), SUM(output_chars) "
            "FROM provider_usage WHERE day >= ? GROUP BY provider, paid "
            "ORDER BY SUM(calls) DESC",
            (time.strftime("%Y-%m-%d", time.localtime(time.time() - days * 86400)),),
        ).fetchall()
        return [
            {"provider": r[0], "paid": bool(r[1]), "calls": r[2],
             "input_chars": r[3], "output_chars": r[4]}
            for r in rows
        ]
    finally:
        conn.close()


def format_usage(days: int = 7) -> str:
    rows = usage_summary(days)
    if not rows:
        return "No AI calls recorded yet."
    lines = [f"💷 *AI usage, last {days} days*\n"]
    for r in rows:
        tag = " 💳 paid" if r["paid"] else ""
        lines.append(
            f"  • *{r['provider']}*{tag} — {r['calls']} calls, "
            f"{round((r['input_chars'] + r['output_chars']) / 1000)}k chars"
        )
    limit = paid_daily_limit()
    used = paid_calls_today()
    lines.append(
        f"\n*Paid calls today:* {used}"
        + (f"/{limit}" + (" — cap reached, paid routes skipped" if used >= limit else "")
           if limit else " (no cap set)")
    )
    return "\n".join(lines)
