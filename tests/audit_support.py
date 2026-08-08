"""
Shared scaffolding for the `test_audit_*` audit suites.

Not a test module (no `test_` prefix, so pytest never collects it). Every
audit test runs against a tmp_path SQLite file with stubbed providers — no
network, no Slack, no real memory.db.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import concept_graph
import memory
import runner


@pytest.fixture
def audit_env(tmp_path, monkeypatch):
    """Isolated DB + a known owner + the critic off unless a test wants it."""
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "memory.db")
    monkeypatch.setattr(concept_graph, "DB_PATH", tmp_path / "memory.db")
    monkeypatch.setattr(runner, "_CALL_AI", None)
    monkeypatch.setattr(runner, "_POST_MESSAGE", None)
    monkeypatch.setenv("OWNER_SLACK_ID", "U_OWNER")
    monkeypatch.setenv("CRITIC_ENABLED", "false")
    monkeypatch.setenv("APPROVALS_ENABLED", "true")
    yield tmp_path


def tool_call(name: str, /, **args) -> str:
    """The text-based tool protocol, exactly as a provider would emit it."""
    return "[TOOL_CALL]\n" + json.dumps({"tool": name, "args": args}) + "\n[/TOOL_CALL]"


class ScriptedAI:
    """Stands in for every AI provider. Records what it was shown."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, messages, prompt=None):
        self.seen.append(list(messages))
        return self.replies.pop(0) if self.replies else "done"

    @property
    def last_transcript(self) -> str:
        return "\n".join(str(m) for m in (self.seen[-1] if self.seen else []))


def run_id_of(row):
    return row["id"] if isinstance(row, dict) else row
