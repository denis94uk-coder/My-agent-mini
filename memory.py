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
from pathlib import Path

logger = logging.getLogger("my-agent-mini")

DB_PATH = Path.home() / "my-agent-mini" / "memory.db"


def get_db() -> sqlite3.Connection:
    """Get a database connection, creating tables if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
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
    """)
    return conn


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
    """Search past conversations by keyword."""
    conn = get_db()
    try:
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


def add_fact(user_id: str, fact: str):
    """Store a fact about a user."""
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO facts (user_id, fact, timestamp) VALUES (?, ?, ?)",
            (user_id, fact, time.time()),
        )
        conn.commit()
        logger.info(f"📝 Stored fact for {user_id}: {fact[:80]}")
    finally:
        conn.close()


def get_facts(user_id: str) -> list[str]:
    """Get all stored facts about a user."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT fact FROM facts WHERE user_id = ? ORDER BY timestamp DESC",
            (user_id,),
        ).fetchall()
        return [r[0] for r in rows]
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
        return cur.rowcount > 0
    finally:
        conn.close()


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
        db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0
        return {
            "messages": msg_count,
            "facts": fact_count,
            "conversations": conv_count,
            "open_tasks": open_tasks,
            "db_size_mb": round(db_size / 1024 / 1024, 2),
        }
    finally:
        conn.close()
