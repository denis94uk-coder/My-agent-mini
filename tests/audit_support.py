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


# ── Slack layer ──
# bot.py is the only module that imports slack_bolt. These helpers import it
# with a recording stand-in for App, so the registered listeners can be driven
# directly: no workspace, no socket, no network.

@pytest.fixture
def slack_bot(audit_env, monkeypatch):
    """Import bot.py with Slack stubbed out. Yields (bot_module, registry)."""
    slack_bolt = pytest.importorskip("slack_bolt")
    registry = {}

    class FakeApp:
        def __init__(self, *a, **k):
            pass

        def _register(self, kind, name):
            def deco(fn):
                registry[(kind, name)] = fn
                return fn
            return deco

        def __getattr__(self, kind):
            return lambda name=None, *a, **k: self._register(kind, name)

    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake-for-tests")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-fake-for-tests")
    monkeypatch.setenv("POLLINATIONS_ENABLED", "false")
    monkeypatch.setattr(slack_bolt, "App", FakeApp)
    sys.modules.pop("bot", None)
    import bot

    class FakeSlackClient:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            def call(**kwargs):
                self.calls.append((name, kwargs))
                return {"ok": True}
            return call

    monkeypatch.setattr(bot, "slack_client", FakeSlackClient())
    monkeypatch.setattr(bot, "call_ai", lambda messages, prompt=None, images=None: "stub answer")
    yield bot, registry
    sys.modules.pop("bot", None)


class Say:
    """Captures what a listener posted back to Slack."""

    def __init__(self):
        self.sent = []

    def __call__(self, text=None, **kw):
        self.sent.append(dict(text=text, **kw))

    @property
    def last(self):
        return self.sent[-1]["text"] if self.sent else None


def slash(registry, command, text="", user="U_STRANGER", channel="C1", **extra):
    """Invoke a registered slash-command listener with a realistic payload."""
    payload = {"command": command, "text": text, "user_id": user,
               "channel_id": channel, "team_id": "T1", "channel_name": "general",
               "response_url": "https://hooks.slack.example/x", "trigger_id": "1.2.3"}
    payload.update(extra)
    say = Say()
    acked = []
    registry[("command", command)](lambda *a, **k: acked.append(True), payload, say)
    return say, acked
