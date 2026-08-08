"""
Phase 3.11 — the Slack layer.

Invariants under test:
  • A slash command acks immediately, whatever the agent then does.
  • A message is handled once. Redelivery, or two event types for the same
    message, must not double-execute the agent.
  • Replies stay in the thread they came from; oversized, empty and unicode
    input are all handled.
  • Nothing accepts an unauthenticated inbound request.

The app runs in Socket Mode (bot.py:1394 `SocketModeHandler`), so there is no
HTTP endpoint and request-signature verification does not apply — that is
asserted rather than assumed, because it is the reason a whole class of test
is absent.

`xfail(strict=True)` marks a CONFIRMED DEFECT — see it fail with
`pytest --runxfail`.
"""

import re
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_support import Say, audit_env, slack_bot, slash  # noqa: F401

import memory


def event(registry, name, payload):
    say = Say()
    registry[("event", name)](payload, say)
    return say


def dm(text="hello", ts="1.1", user="U_USER", **extra):
    return dict({"channel": "D1", "ts": ts, "text": text, "user": user,
                 "channel_type": "im"}, **extra)


# ── ack discipline ──

def test_a_slow_command_still_acks_immediately(audit_env, slack_bot, monkeypatch):
    """Slack gives a slash command 3 seconds. The listener must ack before it
    starts thinking, not after."""
    bot, registry = slack_bot
    order = []

    def slow_ai(messages, prompt=None, images=None):
        order.append("work")
        time.sleep(0.3)
        return "eventually"

    monkeypatch.setattr(bot, "call_ai", slow_ai)
    say = Say()
    payload = {"command": "/ask", "text": "something hard", "user_id": "U_USER",
               "channel_id": "C1", "team_id": "T1"}
    registry[("command", "/ask")](lambda *a, **k: order.append("ack"), payload, say)
    assert order[0] == "ack", f"the agent ran before the ack: {order}"


@pytest.mark.parametrize("command", ["/ask", "/search", "/clear", "/workflow", "/runs",
                                     "/schedules", "/approvals", "/approve", "/deny",
                                     "/costs", "/status", "/health", "/providers"])
def test_every_command_acks(audit_env, slack_bot, command):
    bot, registry = slack_bot
    _, acked = slash(registry, command, user="U_OWNER")
    assert acked, f"{command} never called ack()"


# ── delivered-once ──

def test_a_redelivered_event_is_handled_once(audit_env, slack_bot, monkeypatch):
    """FIXED (was FINDING I3): channel+ts identifies the message, and the
    first listener to claim it wins."""
    bot, registry = slack_bot
    calls = []
    monkeypatch.setattr(bot, "call_ai",
                        lambda m, p=None, images=None: calls.append(1) or "answer")
    payload = dm("expensive question", ts="9.1")
    s1 = event(registry, "message", payload)
    s2 = event(registry, "message", dict(payload))       # Slack redelivers
    assert len(calls) == 1, f"the agent ran {len(calls)} times for one message"
    assert len(s1.sent) + len(s2.sent) == 1


def test_a_mention_inside_a_dm_is_handled_once(audit_env, slack_bot, monkeypatch):
    """FIXED (was FINDING I4): both listeners go through the same claim, so
    whichever event Slack delivers first is the one that runs."""
    bot, registry = slack_bot
    calls = []
    monkeypatch.setattr(bot, "call_ai",
                        lambda m, p=None, images=None: calls.append(1) or "answer")
    payload = dm("<@U0BOT> status?", ts="10.1")
    event(registry, "message", payload)
    event(registry, "app_mention", payload)
    assert len(calls) == 1, f"the agent ran {len(calls)} times for one message"


def test_a_mention_from_another_bot_is_ignored(audit_env, slack_bot, monkeypatch):
    """FIXED (was FINDING I5): handle_mention now has the bot_id guard
    handle_message always had."""
    bot, registry = slack_bot
    calls = []
    monkeypatch.setattr(bot, "call_ai",
                        lambda m, p=None, images=None: calls.append(1) or "answer")
    say = event(registry, "app_mention", {"channel": "C1", "ts": "11.1",
                                          "text": "<@U0BOT> keep talking",
                                          "bot_id": "B999"})
    assert calls == [] and say.sent == []


def test_a_dm_from_another_bot_is_ignored(audit_env, slack_bot, monkeypatch):
    bot, registry = slack_bot
    calls = []
    monkeypatch.setattr(bot, "call_ai",
                        lambda m, p=None, images=None: calls.append(1) or "answer")
    say = event(registry, "message", dm("loop?", ts="12.1", bot_id="B999"))
    assert calls == [] and say.sent == []


def test_channel_chatter_without_a_mention_is_ignored(audit_env, slack_bot):
    bot, registry = slack_bot
    say = event(registry, "message", {"channel": "C1", "ts": "13.1", "text": "chatter",
                                      "user": "U_USER", "channel_type": "channel"})
    assert say.sent == []


def test_a_dm_with_a_file_is_processed(audit_env, slack_bot, monkeypatch):
    """FIXED (was FINDING I6): file_share is no longer filtered out with the
    genuinely noisy subtypes, so the file pipeline is reachable from a DM."""
    bot, registry = slack_bot
    calls = []
    monkeypatch.setattr(bot, "call_ai",
                        lambda m, p=None, images=None: calls.append(1) or "I read it")
    monkeypatch.setattr(bot, "process_slack_files",
                        lambda files: ([], ["📄 *File: notes.txt*\ncontents"]))
    say = event(registry, "message", dm(
        "what's in this?", ts="14.1", subtype="file_share",
        files=[{"name": "notes.txt", "mimetype": "text/plain", "size": 12,
                "url_private_download": "https://files.slack.example/notes.txt"}]))
    assert calls, "the file pipeline was never reached"
    assert say.sent


def test_message_edits_and_deletions_are_ignored(audit_env, slack_bot):
    """The subtype filter is right for these two."""
    bot, registry = slack_bot
    for subtype in ("message_changed", "message_deleted"):
        say = event(registry, "message", dm("edited", ts="15.1", subtype=subtype))
        assert say.sent == [], subtype


# ── threading and rendering ──

def test_a_threaded_mention_replies_in_that_thread(audit_env, slack_bot):
    bot, registry = slack_bot
    say = event(registry, "app_mention", {"channel": "C1", "ts": "16.9",
                                          "thread_ts": "16.1", "user": "U_USER",
                                          "text": "<@U0BOT> in thread"})
    assert say.sent[-1].get("thread_ts") == "16.1"


def test_a_dm_reply_threads_under_the_message(audit_env, slack_bot):
    bot, registry = slack_bot
    say = event(registry, "message", dm("hi", ts="17.1"))
    assert say.sent[-1].get("thread_ts") == "17.1"


def test_the_loading_reaction_lands_on_the_users_message(audit_env, slack_bot):
    """Not on the thread root — that would mark the wrong message."""
    bot, registry = slack_bot
    event(registry, "app_mention", {"channel": "C1", "ts": "18.9", "thread_ts": "18.1",
                                    "user": "U_USER", "text": "<@U0BOT> hi"})
    stamps = [kw.get("timestamp") for name, kw in bot.slack_client.calls
              if name.startswith("reactions")]
    assert stamps and set(stamps) == {"18.9"}


def test_the_loading_reaction_is_removed_even_when_the_agent_fails(audit_env, slack_bot, monkeypatch):
    bot, registry = slack_bot

    def boom(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(bot, "call_ai", boom)
    event(registry, "message", dm("hi", ts="19.1"))
    removed = [kw for name, kw in bot.slack_client.calls if name == "reactions_remove"]
    assert removed, "the hourglass would sit on the message forever"


def test_a_long_reply_is_truncated_for_slack(audit_env, slack_bot, monkeypatch):
    bot, registry = slack_bot
    monkeypatch.setattr(bot, "call_ai", lambda m, p=None, images=None: "y" * 9000)
    say = event(registry, "message", dm("write me an essay", ts="20.1"))
    assert len(say.last) <= 4000
    assert "truncated" in say.last


def test_unicode_and_emoji_survive_the_round_trip(audit_env, slack_bot, monkeypatch):
    bot, registry = slack_bot
    monkeypatch.setattr(bot, "call_ai", lambda m, p=None, images=None: "réponse ✅ 日本語")
    say = event(registry, "message", dm("café ☕ 日本語?", ts="21.1"))
    assert say.last == "réponse ✅ 日本語"
    assert "café ☕ 日本語?" in memory.get_history("D1:21.1", limit=5)[0]["content"]


def test_an_empty_ask_gives_usage_not_an_ai_call(audit_env, slack_bot, monkeypatch):
    bot, registry = slack_bot
    calls = []
    monkeypatch.setattr(bot, "call_ai", lambda m, p=None, images=None: calls.append(1) or "x")
    for text in ("", "   ", "\n"):
        say, _ = slash(registry, "/ask", text=text)
        assert (say.last or "").startswith("Usage:")
    assert calls == []


def test_a_bare_mention_gets_a_greeting_not_an_ai_call(audit_env, slack_bot, monkeypatch):
    bot, registry = slack_bot
    calls = []
    monkeypatch.setattr(bot, "call_ai", lambda m, p=None, images=None: calls.append(1) or "x")
    say = event(registry, "app_mention", {"channel": "C1", "ts": "22.1",
                                          "user": "U_USER", "text": "<@U0BOT>"})
    assert "Hey!" in say.last and calls == []


def test_the_labelled_mention_form_is_stripped(audit_env, slack_bot):
    """FIXED (was FINDING I7): the regex matches Slack's labelled mention form
    too, so it never reaches the model or memory as raw markup."""
    bot, registry = slack_bot
    event(registry, "app_mention", {"channel": "C1", "ts": "23.1", "user": "U_USER",
                                    "text": "<@U0BOT|agent> summarise"})
    stored = memory.get_history("C1:23.1", limit=5)[0]["content"]
    assert "<@" not in stored, f"raw mention markup stored: {stored!r}"


def test_a_malformed_slash_payload_does_not_raise(audit_env, slack_bot):
    """FIXED (was FINDING I8): commands read the channel through a helper that
    degrades instead of raising KeyError into a logged 500."""
    bot, registry = slack_bot
    say = Say()
    registry[("command", "/ask")](lambda *a, **k: None,
                                  {"command": "/ask", "text": "hi", "user_id": "U_USER"},
                                  say)


# ── inbound authentication ──

def test_there_is_no_unauthenticated_inbound_surface(audit_env):
    """Socket Mode: the app dials out over an authenticated WebSocket. There is
    no HTTP endpoint, so there is no unsigned request to reject — and no
    signing secret to misconfigure."""
    source = Path(__file__).resolve().parent.parent.joinpath("bot.py").read_text()
    for marker in ("flask", "fastapi", "Flask", "FastAPI", "http.server",
                   "/slack/events", "signing_secret", "SLACK_SIGNING_SECRET"):
        assert marker not in source, f"an HTTP surface appeared: {marker}"
    assert "SocketModeHandler" in source


def test_the_app_level_token_is_required_to_start(audit_env):
    source = Path(__file__).resolve().parent.parent.joinpath("bot.py").read_text()
    assert "if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:" in source
    assert "raise SystemExit(1)" in source


def test_outbound_posts_retry_on_a_slack_429(audit_env, slack_bot):
    """FIXED (was FINDING I9): slack_sdk's default handlers cover connection
    errors only, so a Slack 429 raised and runner._notify dropped a finished
    run's result. The client bot.py builds now carries a rate-limit handler."""
    import importlib
    import inspect

    bot, _ = slack_bot
    source = inspect.getsource(importlib.import_module("bot"))
    assert "RateLimitErrorRetryHandler" in source, (
        "the WebClient is built without a rate-limit retry handler")

    from slack_sdk import WebClient
    from slack_sdk.http_retry import default_retry_handlers
    from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler

    client = WebClient(
        token="xoxb-x",
        retry_handlers=default_retry_handlers() + [RateLimitErrorRetryHandler(max_retry_count=3)],
    )
    assert any("RateLimit" in type(h).__name__ for h in client.retry_handlers)


def test_a_failed_post_never_kills_the_listener(audit_env, slack_bot, monkeypatch):
    """FIXED (was FINDING I10): the apology post has its own try/except, so a
    channel the bot cannot post to produces one log line, not a traceback."""
    """say() raising (not_in_channel, Slack down) must not escape the listener."""
    bot, registry = slack_bot
    monkeypatch.setattr(bot, "call_ai", lambda m, p=None, images=None: "answer")

    def failing_say(text=None, **kw):
        raise RuntimeError("slack_api_error: not_in_channel")

    registry[("event", "message")](dm("hi", ts="24.1"), failing_say)
