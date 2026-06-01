"""
Persistent conversation memory using SQLite.
Stores conversations and user facts to disk — survives restarts.
Uses ~1-2 MB of RAM regardless of history size.
"""

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


def clear_conversation(conv_key: str):
    """Clear history for a specific conversation."""
    conn = get_db()
    try:
        conn.execute("DELETE FROM conversations WHERE conv_key = ?", (conv_key,))
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
        db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0
        return {
            "messages": msg_count,
            "facts": fact_count,
            "conversations": conv_count,
            "db_size_mb": round(db_size / 1024 / 1024, 2),
        }
    finally:
        conn.close()
