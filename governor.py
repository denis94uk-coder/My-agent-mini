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
import re
import logging
import threading

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
    "github_list_pull_requests": READ,
    "github_pr_status": READ,
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


# Owner-only and risk tier are two different axes, and conflating them is a
# mistake worth naming. OWNER_ONLY_TOOLS answers "which human may invoke
# this" — run_shell is owner-only because it is remote code execution on the
# host. The tier answers "how far does the effect reach" — run_shell is
# WRITE_LOCAL because it stays on this box.
#
# The invariant that actually matters runs one way only: anything EXTERNAL
# must also be owner-only. The converse is false, and asserting it would force
# `ls` in a scheduled task to raise a Slack approval — which is precisely how
# an approval queue degrades into a button people press without reading.
def external_tools() -> set:
    """Tools whose effects leave this server. These need per-call approval."""
    return {name for name, tier in TOOL_TIERS.items() if tier == EXTERNAL}


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


# ── Rate-limit awareness ──
#
# Free routes are capped on tokens per minute long before tokens per day, and
# the agent loop is the exact shape that trips it: up to MAX_ITERATIONS calls
# fired back to back, each re-sending the whole system prompt. Groq's free
# tier allows 12,000 TPM against a ~4,400-token prompt, so the third call of
# any multi-step task is rejected while ~95% of the daily budget is untouched.
#
# Waiting a few seconds for the window to roll is strictly better than
# spending a 429 to discover the same thing, so the router asks first.
# Tracking is in-memory: a minute window has nothing worth persisting, and it
# must stay cheap on a 1 GB box.

_TPM_DEFAULTS = {"groq": 12000, "gemini": 250000}
# Requests per minute is a separate ceiling and can bind first: Gemini's free
# tier allows ~10 requests/minute against a 250,000 token/minute budget, so a
# ten-step agent loop exhausts the request limit having spent 2% of the tokens.
# Tracking only tokens would pace perfectly and still 429.
_RPM_DEFAULTS = {"groq": 30, "gemini": 10}
_TPM_WINDOW: dict[str, list] = {}
_TPM_LOCK = threading.Lock()

# Output length is unknown before the call. Charge a flat allowance so the
# estimate errs high — undercounting spends a 429, overcounting costs a pause.
_OUTPUT_TOKEN_ALLOWANCE = 800


def _limit_for(provider_name: str, defaults: dict, env_var: str) -> int:
    """Per-route ceiling from `env_var`, falling back to `defaults`, else 0."""
    lowered = (provider_name or "").lower()
    table = dict(defaults)
    for pair in os.getenv(env_var, "").split(","):
        name, _, value = pair.partition(":")
        if name.strip() and value.strip().isdigit():
            table[name.strip().lower()] = int(value)
    for key, limit in table.items():
        if key in lowered:
            return max(0, limit)
    return 0


def tpm_limit(provider_name: str) -> int:
    """
    Tokens-per-minute ceiling for a route, or 0 for "unmetered".

    ROUTER_TPM_LIMITS overrides the defaults as `substring:tokens` pairs,
    e.g. "groq:12000,gemini:250000". Matching is by substring so it keeps
    working when a provider is renamed in build_providers.
    """
    return _limit_for(provider_name, _TPM_DEFAULTS, "ROUTER_TPM_LIMITS")


def rpm_limit(provider_name: str) -> int:
    """Requests-per-minute ceiling, or 0 for "unmetered". See ROUTER_RPM_LIMITS."""
    return _limit_for(provider_name, _RPM_DEFAULTS, "ROUTER_RPM_LIMITS")


def requests_last_minute(provider_name: str) -> int:
    now = time.time()
    with _TPM_LOCK:
        return sum(1 for ts, _ in _TPM_WINDOW.get(provider_name, []) if now - ts < 60)


def retry_after_seconds(response, default: float) -> float:
    """
    How long a rate-limited route asked us to wait, in seconds.

    Reads Retry-After (seconds, per RFC 9110) and falls back to the
    x-ratelimit-reset-* headers OpenAI-compatible providers send, which carry
    durations like "7.66s" or "2m59.56s" rather than plain numbers. Returns
    `default` when the response says nothing usable, so a missing header is
    never read as "retry immediately".

    Capped, because one header should not be able to park a route for an hour.
    """
    headers = getattr(response, "headers", None) or {}
    for key in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        try:
            raw = (headers.get(key) or "").strip()
        except Exception:
            continue
        if not raw:
            continue
        # Retry-After also permits an HTTP-date. Its digits look enough like a
        # duration that the parser below would read "Wed, 21 Oct 2015 07:28:00
        # GMT" as 2,071 seconds, so date form is handled before that can
        # happen — and a date already in the past means retry now.
        if "," in raw or "GMT" in raw.upper():
            try:
                from email.utils import parsedate_to_datetime
                delta = parsedate_to_datetime(raw).timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                continue
            return min(max(delta, 0.0), max(default, 300)) if delta > 0 else 0.0
        seconds, matched = 0.0, False
        for value, unit in re.findall(r"([\d.]+)\s*(ms|s|m|h)?", raw):
            try:
                number = float(value)
            except ValueError:
                continue
            seconds += number * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}.get(unit or "s", 1)
            matched = True
        if matched and seconds > 0:
            return min(seconds + 1, max(default, 300))
    return default


_UNRANKED = 10_000


def route_order_rank(provider_name: str) -> int:
    """
    Where a route sits in ROUTER_ORDER, or `_UNRANKED` when it is not listed.

    Registration order in build_providers is just the order the blocks happen
    to be written in, which is not a preference anyone chose — it put Gemini
    ahead of Groq purely because its block comes first. ROUTER_ORDER makes the
    preference explicit: "groq,gemini" tries Groq first, and anything omitted
    keeps its registration order behind everything named.

    Matching is by substring, so "groq" finds "Groq" and a renamed route keeps
    working. Paid routes still sort last regardless — that is a budget guard,
    not a preference, and it is not this function's to override.
    """
    lowered = (provider_name or "").lower()
    names = [n.strip().lower() for n in os.getenv("ROUTER_ORDER", "").split(",") if n.strip()]
    for index, name in enumerate(names):
        if name in lowered:
            return index
    return _UNRANKED


def record_tokens(provider_name: str, tokens: int) -> None:
    """Add a call's token cost to the rolling minute window."""
    now = time.time()
    with _TPM_LOCK:
        window = [e for e in _TPM_WINDOW.get(provider_name, []) if now - e[0] < 60]
        window.append((now, max(0, int(tokens))))
        _TPM_WINDOW[provider_name] = window


def tokens_last_minute(provider_name: str) -> int:
    now = time.time()
    with _TPM_LOCK:
        return sum(t for ts, t in _TPM_WINDOW.get(provider_name, []) if now - ts < 60)


def _window_wait(window: list, now: float, limit: int, cost, incoming: int) -> float:
    """Seconds until `incoming` more of `cost` fits under `limit`."""
    spent = sum(cost(e) for e in window)
    if spent + incoming <= limit:
        return 0.0
    freed = 0
    for entry in window:  # oldest first; each expiry frees its own share
        freed += cost(entry)
        if spent - freed + incoming <= limit:
            return max(0.0, 60 - (now - entry[0])) + 0.5
    return 0.0


def tpm_wait_seconds(provider_name: str, est_tokens: int) -> float:
    """
    Seconds to wait before this call fits inside the route's minute window.

    Covers both ceilings. Tokens alone are not enough: Gemini's free tier
    pairs a generous 250,000 tokens/minute with ~10 requests/minute, so an
    agent loop exhausts the request limit having spent 2% of the tokens, and
    token-only pacing would wait for nothing and still 429.

    0 when it fits now or the route is unmetered. A single call larger than
    the whole token limit also returns 0 — waiting cannot help, and the
    honest 429 is more useful than a pause that changes nothing.
    """
    tokens, requests = tpm_limit(provider_name), rpm_limit(provider_name)
    if not tokens and not requests:
        return 0.0
    if tokens and est_tokens >= tokens:
        return 0.0
    now = time.time()
    with _TPM_LOCK:
        window = sorted(e for e in _TPM_WINDOW.get(provider_name, []) if now - e[0] < 60)
    waits = [0.0]
    if tokens:
        waits.append(_window_wait(window, now, tokens, lambda e: e[1], est_tokens))
    if requests:
        waits.append(_window_wait(window, now, requests, lambda e: 1, 1))
    return max(waits)


def estimate_tokens(input_chars: int) -> int:
    """Rough token cost of a call. ~4 chars/token plus an output allowance."""
    return max(0, input_chars) // 4 + _OUTPUT_TOKEN_ALLOWANCE


def reset_rate_limit_state() -> None:
    """Test hook — the minute window is process state, not database state."""
    with _TPM_LOCK:
        _TPM_WINDOW.clear()


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
