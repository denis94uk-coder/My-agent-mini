"""
Persistent conversation memory using SQLite.
Stores conversations and user facts to disk — survives restarts.
Uses ~1-2 MB of RAM regardless of history size.
"""

import re
import sqlite3
import json
import time
import logging
import threading
from pathlib import Path

logger = logging.getLogger("my-agent-mini")

DB_PATH = Path.home() / "my-agent-mini" / "memory.db"


# Schema creation is a write transaction. Running it on every `get_db()` call
# meant every read opened by taking a write lock, so the workers, the scheduler
# and the sweeper serialised against each other on a file they were only
# reading — and under real concurrency that surfaced as 'database is locked'.
# The schema is created once per database path per process instead. Keyed on
# the path rather than a bare flag because the tests point DB_PATH at a tmpdir,
# and a process-wide flag would leave those databases with no tables at all.
_SCHEMA_READY: set[str] = set()
_SCHEMA_LOCK = threading.Lock()

# How long a writer waits for a competing write to finish before giving up.
# SQLite's default is 5s, which is generous for a single statement and far too
# short for a queue drain behind a slow one; the failure mode is a lost run.
_BUSY_TIMEOUT_SECONDS = 30.0


def get_db() -> sqlite3.Connection:
    """Get a database connection, creating tables if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Check for the file *before* connecting, because connecting creates it.
    # Without this, a database deleted out from under a running process would
    # be reopened empty and never re-created — the old code got this right by
    # accident, by running the schema every single time.
    db_existed = DB_PATH.exists()
    conn = sqlite3.connect(str(DB_PATH), timeout=_BUSY_TIMEOUT_SECONDS)
    # `timeout` above only covers the Python driver's own retry loop; statements
    # SQLite runs internally honour busy_timeout instead. Set both.
    conn.execute(f"PRAGMA busy_timeout={int(_BUSY_TIMEOUT_SECONDS * 1000)}")

    path_key = str(DB_PATH)
    if db_existed and path_key in _SCHEMA_READY:
        return conn

    with _SCHEMA_LOCK:
        if db_existed and path_key in _SCHEMA_READY:
            return conn
        _ensure_schema(conn)
        _SCHEMA_READY.add(path_key)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables and indexes. Runs once per database path per process."""
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conv_key TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_conv_key ON conversations(conv_key);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp);

        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            fact TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'fact',
            timestamp REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_user ON facts(user_id);

        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conv_key TEXT NOT NULL,
            user_id TEXT NOT NULL,
            step_no INTEGER NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created REAL NOT NULL,
            updated REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_task_conv ON tasks(conv_key);

        CREATE TABLE IF NOT EXISTS thread_summaries (
            conv_key TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            message_count INTEGER NOT NULL DEFAULT 0,
            updated REAL NOT NULL
        );
    """)
    _ensure_fts(conn)
    # Migration: older DBs may have a `facts` table from before the
    # `category` column existed. Add it in place so existing installs
    # don't need to wipe memory.db to pick up durable project memory.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(facts)").fetchall()}
    if "category" not in existing_cols:
        conn.execute("ALTER TABLE facts ADD COLUMN category TEXT NOT NULL DEFAULT 'fact'")
        conn.commit()
    conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON facts(category)")
    conn.commit()


# ── Full-text search (FTS5) ──
# SQLite ships FTS5 on Ubuntu's python3, so we get real ranked full-text
# search (BM25) over the whole conversation history for free — no
# embeddings, no extra RAM. If this SQLite build lacks FTS5 we silently
# fall back to LIKE-based search everywhere.

FTS_AVAILABLE = True


def _ensure_fts(conn: sqlite3.Connection) -> None:
    """Create the FTS index + sync triggers; backfill existing rows once."""
    global FTS_AVAILABLE
    try:
        conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS conv_fts USING fts5(
                content, content='conversations', content_rowid='id'
            );
            CREATE TRIGGER IF NOT EXISTS conv_fts_ai AFTER INSERT ON conversations BEGIN
                INSERT INTO conv_fts(rowid, content) VALUES (new.id, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS conv_fts_ad AFTER DELETE ON conversations BEGIN
                INSERT INTO conv_fts(conv_fts, rowid, content) VALUES ('delete', old.id, old.content);
            END;
        """)
        # One-time backfill for rows inserted before the index existed.
        fts_rows = conn.execute("SELECT COUNT(*) FROM conv_fts").fetchone()[0]
        conv_rows = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        if fts_rows < conv_rows:
            conn.execute(
                "INSERT INTO conv_fts(rowid, content) "
                "SELECT id, content FROM conversations "
                "WHERE id NOT IN (SELECT rowid FROM conv_fts)"
            )
        conn.commit()
        FTS_AVAILABLE = True
    except sqlite3.OperationalError as e:
        FTS_AVAILABLE = False
        logger.warning(f"FTS5 unavailable, falling back to LIKE search: {e}")


def _fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 OR-query of quoted terms."""
    terms = [w for w in re.findall(r"[A-Za-z0-9']{3,}", text) if w.lower() not in _STOPWORDS]
    return " OR ".join(f'"{t}"' for t in terms[:12])


def add_message(conv_key: str, role: str, content: str):
    """Store a message in conversation history."""
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO conversations (conv_key, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (conv_key, role, content, time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def get_history(conv_key: str, limit: int = 20) -> list[dict]:
    """Get recent conversation history for a thread."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE conv_key = ? ORDER BY id DESC LIMIT ?",
            (conv_key, limit),
        ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    finally:
        conn.close()


def search_history(query: str, limit: int = 10) -> list[dict]:
    """Search past conversations. Ranked FTS5 (BM25) when available, LIKE fallback."""
    conn = get_db()
    try:
        if FTS_AVAILABLE:
            fts_q = _fts_query(query)
            if fts_q:
                try:
                    rows = conn.execute(
                        "SELECT c.conv_key, c.role, c.content, c.timestamp "
                        "FROM conv_fts f JOIN conversations c ON c.id = f.rowid "
                        "WHERE conv_fts MATCH ? ORDER BY bm25(conv_fts) LIMIT ?",
                        (fts_q, limit),
                    ).fetchall()
                    return [
                        {"conv_key": r[0], "role": r[1], "content": r[2][:300], "time": r[3]}
                        for r in rows
                    ]
                except sqlite3.OperationalError:
                    pass  # malformed query → fall through to LIKE
        rows = conn.execute(
            "SELECT conv_key, role, content, timestamp FROM conversations "
            "WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [
            {"conv_key": r[0], "role": r[1], "content": r[2][:300], "time": r[3]}
            for r in rows
        ]
    finally:
        conn.close()


def search_all_relevant(query: str, exclude_conv_key: str = "", limit: int = 4) -> list[dict]:
    """
    Cross-thread memory: find messages from OTHER conversations relevant to
    `query` — this is what lets a brand-new Slack thread recall decisions and
    discussions from weeks ago. Ranked by BM25 with a mild recency boost.
    Also searches saved thread summaries so long-dead threads surface as one
    compact digest instead of raw messages.
    """
    results: list[dict] = []
    conn = get_db()
    try:
        # 1. Thread summaries (compact, high-signal) — LIKE on keywords.
        for kw in list(_keywords(query))[:6]:
            rows = conn.execute(
                "SELECT conv_key, summary, updated FROM thread_summaries "
                "WHERE conv_key != ? AND summary LIKE ? LIMIT 3",
                (exclude_conv_key, f"%{kw}%"),
            ).fetchall()
            for r in rows:
                if not any(x.get("conv_key") == r[0] and x["kind"] == "summary" for x in results):
                    results.append({"kind": "summary", "conv_key": r[0], "content": r[1][:500], "time": r[2]})

        # 2. Raw messages from other threads via FTS.
        if FTS_AVAILABLE:
            fts_q = _fts_query(query)
            if fts_q:
                try:
                    now = time.time()
                    rows = conn.execute(
                        "SELECT c.conv_key, c.role, c.content, c.timestamp, bm25(conv_fts) AS rank "
                        "FROM conv_fts f JOIN conversations c ON c.id = f.rowid "
                        "WHERE conv_fts MATCH ? AND c.conv_key != ? "
                        "ORDER BY rank LIMIT 20",
                        (fts_q, exclude_conv_key),
                    ).fetchall()
                    scored = []
                    for conv_key, role, content, ts, rank in rows:
                        age_days = max(0.0, (now - ts) / 86400)
                        score = -rank - min(age_days * 0.05, 3.0)  # bm25 is negative-better
                        scored.append((score, conv_key, role, content, ts))
                    scored.sort(reverse=True)
                    for _, conv_key, role, content, ts in scored:
                        if len(results) >= limit + 2:
                            break
                        results.append({"kind": "message", "conv_key": conv_key, "role": role,
                                        "content": content[:350], "time": ts})
                except sqlite3.OperationalError:
                    pass
    finally:
        conn.close()
    return results[: limit + 2]


# ── Thread summaries (durable cross-thread memory) ──

def get_summary_state(conv_key: str) -> tuple[str, int]:
    """Return (existing summary, message_count at last summary) for a thread."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT summary, message_count FROM thread_summaries WHERE conv_key = ?",
            (conv_key,),
        ).fetchone()
        return (row[0], row[1]) if row else ("", 0)
    finally:
        conn.close()


def count_messages(conv_key: str) -> int:
    conn = get_db()
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE conv_key = ?", (conv_key,)
        ).fetchone()[0]
    finally:
        conn.close()


def save_thread_summary(conv_key: str, user_id: str, summary: str, message_count: int):
    """Upsert the rolling summary for a thread."""
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO thread_summaries (conv_key, user_id, summary, message_count, updated) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(conv_key) DO UPDATE SET summary=excluded.summary, "
            "message_count=excluded.message_count, updated=excluded.updated",
            (conv_key, user_id, summary.strip()[:2000], message_count, time.time()),
        )
        conn.commit()
        logger.info(f"🧾 Saved thread summary for {conv_key} ({message_count} msgs)")
    finally:
        conn.close()


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of",
    "and", "or", "in", "on", "at", "for", "with", "this", "that", "it",
    "i", "you", "he", "she", "we", "they", "my", "your", "me", "do", "does",
    "did", "can", "could", "would", "should", "will", "what", "how", "why",
    "please", "hey", "hi", "hello",
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']{4,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def search_relevant(conv_key: str, query: str, exclude_recent: int = 20, limit: int = 5) -> list[dict]:
    """
    Find older messages in this conversation that are topically relevant to
    `query` but fall outside the most recent `exclude_recent` messages (which
    are already included via get_history). Cheap keyword-overlap scoring —
    no embeddings, so it stays fast and free on a 1 GB VM.
    """
    q_words = _keywords(query)
    if not q_words:
        return []

    conn = get_db()
    try:
        # Grab a bounded window of older messages to score, oldest excluded
        # by simple id-based windowing rather than loading the whole table.
        recent_ids = [
            r[0] for r in conn.execute(
                "SELECT id FROM conversations WHERE conv_key = ? ORDER BY id DESC LIMIT ?",
                (conv_key, exclude_recent),
            ).fetchall()
        ]
        min_recent_id = min(recent_ids) if recent_ids else 0
        rows = conn.execute(
            "SELECT role, content, timestamp FROM conversations "
            "WHERE conv_key = ? AND id < ? ORDER BY id DESC LIMIT 300",
            (conv_key, min_recent_id),
        ).fetchall()
    finally:
        conn.close()

    scored = []
    for role, content, ts in rows:
        c_words = _keywords(content)
        overlap = q_words & c_words
        if overlap:
            scored.append((len(overlap), role, content, ts))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"role": role, "content": content[:400], "time": ts}
        for _, role, content, ts in scored[:limit]
    ]


def add_fact(user_id: str, fact: str, category: str = "fact"):
    """
    Store a fact about a user, tagged by category:
      - 'fact'         casual/ambient info (preferences, small details)
      - 'decision'     durable project decisions, stated priorities, roadmap
                       items — things that must survive into future threads
      - 'task_summary' auto-generated digest of a completed task/plan
    Category matters for retrieval: decisions and task_summaries are treated
    as durable project memory and never get crowded out the way plain
    'fact' entries can (see get_facts).
    """
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO facts (user_id, fact, category, timestamp) VALUES (?, ?, ?, ?)",
            (user_id, fact, category, time.time()),
        )
        conn.commit()
        logger.info(f"📝 Stored {category} for {user_id}: {fact[:80]}")
    finally:
        conn.close()


def get_facts(user_id: str, recent_fact_limit: int = 10, decision_limit: int = 40) -> dict[str, list[str]]:
    """
    Get durable project memory for a user, split into two groups so the
    caller can render them separately and neither crowds the other out:
      - 'durable': ALL 'decision' and 'task_summary' entries (up to
        decision_limit), newest first — must persist across threads.
      - 'recent': the most recent `recent_fact_limit` plain 'fact' entries —
        ambient preferences, capped so they don't grow the prompt unboundedly.
    """
    conn = get_db()
    try:
        durable = conn.execute(
            "SELECT fact FROM facts WHERE user_id = ? AND category IN ('decision', 'task_summary') "
            "ORDER BY timestamp DESC LIMIT ?",
            (user_id, decision_limit),
        ).fetchall()
        recent = conn.execute(
            "SELECT fact FROM facts WHERE user_id = ? AND category = 'fact' "
            "ORDER BY timestamp DESC LIMIT ?",
            (user_id, recent_fact_limit),
        ).fetchall()
        return {"durable": [r[0] for r in durable], "recent": [r[0] for r in recent]}
    finally:
        conn.close()


# ── Task Planner ──
# Lightweight persistent to-do list per conversation. The agent creates a
# plan for multi-step work, then checks steps off as it completes them —
# gives the user visibility and gives the agent a durable memory of where
# it left off if a task spans multiple turns.

def create_plan(conv_key: str, user_id: str, steps: list[str]) -> list[dict]:
    """Replace any existing plan for this conversation with new steps."""
    conn = get_db()
    try:
        conn.execute("DELETE FROM tasks WHERE conv_key = ?", (conv_key,))
        now = time.time()
        for i, step in enumerate(steps, 1):
            conn.execute(
                "INSERT INTO tasks (conv_key, user_id, step_no, description, status, created, updated) "
                "VALUES (?, ?, ?, ?, 'pending', ?, ?)",
                (conv_key, user_id, i, step.strip(), now, now),
            )
        conn.commit()
    finally:
        conn.close()
    return get_plan(conv_key)


def get_plan(conv_key: str) -> list[dict]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT step_no, description, status FROM tasks WHERE conv_key = ? ORDER BY step_no",
            (conv_key,),
        ).fetchall()
        return [{"step_no": r[0], "description": r[1], "status": r[2]} for r in rows]
    finally:
        conn.close()


def update_task_status(conv_key: str, step_no: int, status: str) -> bool:
    conn = get_db()
    try:
        cur = conn.execute(
            "UPDATE tasks SET status = ?, updated = ? WHERE conv_key = ? AND step_no = ?",
            (status, time.time(), conv_key, step_no),
        )
        conn.commit()
        updated = cur.rowcount > 0
    finally:
        conn.close()

    if updated and status == "done":
        _maybe_summarize_completed_plan(conv_key)
    return updated


def _maybe_summarize_completed_plan(conv_key: str):
    """
    If every step in this conversation's plan is now 'done', write a durable
    task_summary fact automatically — so finished work is remembered in
    future threads even if the model never explicitly calls `remember`.
    Deterministic digest of the plan's own step descriptions, no extra LLM
    call needed.
    """
    plan = get_plan(conv_key)
    if not plan or any(p["status"] != "done" for p in plan):
        return
    steps_text = "; ".join(p["description"] for p in plan)
    summary = f"[Completed task in {conv_key}] {steps_text}"
    # user_id isn't tracked per-task-row for lookup here, so re-derive it
    # from the tasks table where it was stored at plan-creation time.
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT user_id FROM tasks WHERE conv_key = ? LIMIT 1", (conv_key,)
        ).fetchone()
    finally:
        conn.close()
    user_id = row[0] if row else "default"
    add_fact(user_id, summary, category="task_summary")


def clear_conversation(conv_key: str):
    """Clear history and any pending task plan for a specific conversation."""
    conn = get_db()
    try:
        conn.execute("DELETE FROM conversations WHERE conv_key = ?", (conv_key,))
        conn.execute("DELETE FROM tasks WHERE conv_key = ?", (conv_key,))
        conn.commit()
    finally:
        conn.close()


def get_stats() -> dict:
    """Get memory statistics."""
    conn = get_db()
    try:
        msg_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        fact_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        conv_count = conn.execute("SELECT COUNT(DISTINCT conv_key) FROM conversations").fetchone()[0]
        open_tasks = conn.execute("SELECT COUNT(*) FROM tasks WHERE status != 'done'").fetchone()[0]
        summaries = conn.execute("SELECT COUNT(*) FROM thread_summaries").fetchone()[0]
        db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0
        return {
            "messages": msg_count,
            "facts": fact_count,
            "conversations": conv_count,
            "open_tasks": open_tasks,
            "thread_summaries": summaries,
            "fts_enabled": FTS_AVAILABLE,
            "db_size_mb": round(db_size / 1024 / 1024, 2),
        }
    finally:
        conn.close()
