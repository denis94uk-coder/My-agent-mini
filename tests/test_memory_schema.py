"""
Schema setup and write contention on the shared SQLite file.

`get_db()` used to run `PRAGMA journal_mode=WAL` plus the whole `CREATE TABLE
IF NOT EXISTS` script on every single call. Both are write transactions, so
*every* connection — including read-only ones — opened by taking a write lock.
Two workers, the scheduler and the stalled-plan sweeper all share one file, and
under real concurrency that surfaced as `database is locked`, which loses a run.

The schema now runs once per database path per process. These tests pin the
three things that made the old behaviour accidentally correct, so the
optimisation cannot quietly break them.
"""

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memory


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "memory.db")
    memory._SCHEMA_READY.discard(str(tmp_path / "memory.db"))
    yield


def test_schema_is_created_on_first_use():
    memory.add_message("c", "user", "hello")
    assert memory.get_history("c") == [{"role": "user", "content": "hello"}]


def test_schema_runs_once_not_on_every_connection(monkeypatch):
    """The point of the change. If this regresses, the lock contention comes
    back and only shows up as a flaky failure under load."""
    calls = []
    real = memory._ensure_schema
    monkeypatch.setattr(
        memory, "_ensure_schema",
        lambda conn: (calls.append(1), real(conn))[1],
    )

    for _ in range(5):
        memory.get_db().close()

    assert len(calls) == 1, f"schema ran {len(calls)} times, expected 1"


def test_a_second_database_path_gets_its_own_schema(tmp_path, monkeypatch):
    """INVARIANT: the guard is keyed on the path, not a process-wide flag.
    A bare flag would leave every test's tmpdir database with no tables."""
    memory.get_db().close()

    other = tmp_path / "other.db"
    monkeypatch.setattr(memory, "DB_PATH", other)
    memory.add_message("c", "user", "in the other db")

    assert memory.get_history("c") == [{"role": "user", "content": "in the other db"}]


def test_a_deleted_database_is_rebuilt():
    """The old code re-created the schema on every call, so deleting memory.db
    under a running process self-healed. Caching must not lose that."""
    memory.add_message("c", "user", "before")
    memory.DB_PATH.unlink()

    memory.add_message("c", "user", "after")

    assert memory.get_history("c") == [{"role": "user", "content": "after"}]


def test_busy_timeout_is_set_on_the_connection():
    """`sqlite3.connect(timeout=...)` only covers the driver's own retry loop;
    statements SQLite runs internally consult busy_timeout instead."""
    conn = memory.get_db()
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 30000
    finally:
        conn.close()


def test_wal_is_still_enabled():
    conn = memory.get_db()
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


def test_concurrent_writers_do_not_hit_database_is_locked():
    """The regression this fix is for. Deliberately heavier than the copy in
    test_audit_autonomy.py, which passed by luck depending on test ordering."""
    errors = []

    def writer(n):
        try:
            for i in range(60):
                memory.add_message(f"C{n}", "user", f"m{i}")
                memory.search_history("m")
        except Exception as e:  # pragma: no cover - only on contention
            errors.append(repr(e))

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert not errors, errors
