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

@pytest.mark.xfail(strict=True, reason=(
    "FINDING I3 (MEDIUM) — no delivery deduplication. bot.py's message and "
    "app_mention listeners (bot.py:963, bot.py:997) key on nothing: there is no "
    "seen-event table and no check of event ts / event_id / "
    "x-slack-retry-num, which slack_bolt does surface in socket mode "
    "(slack_bolt/adapter/socket_mode/internals.py:20). A redelivery after a "
    "socket reconnect therefore runs the whole agent loop again: a second "
    "reply, a second set of tool calls, double the AI spend. Bolt acks events "
    "immediately (thread_runner.py:106), so this is a reconnect-window bug "
    "rather than a per-message one — but the same gap also means two event "
    "types for one message are processed twice (see FINDING I4)."))
def test_a_redelivered_event_is_handled_once(audit_env, slack_bot, monkeypatch):
    bot, registry = slack_bot
    calls = []
    monkeypatch.setattr(bot, "call_ai",
                        lambda m, p=None, images=None: calls.append(1) or "answer")
    payload = dm("expensive question", ts="9.1")
    s1 = event(registry, "message", payload)
    s2 = event(registry, "message", dict(payload))       # Slack redelivers
    assert len(calls) == 1, f"the agent ran {len(calls)} times for one message"
    assert len(s1.sent) + len(s2.sent) == 1


@pytest.mark.xfail(strict=True, reason=(
    "FINDING I4 (MEDIUM) — an @mention inside a DM is handled twice. Slack "
    "delivers both message.im and app_mention for it; handle_message "
    "(bot.py:963) filters only on bot_id, subtype and channel_type, and never "
    "checks whether the text mentions the bot, while handle_mention "
    "(bot.py:997) has no channel_type filter. Both therefore run the full "
    "agent loop: two replies, two sets of tool calls. Needs a live workspace "
    "to confirm Slack's delivery of app_mention in IMs; the code-level gap — "
    "neither listener excludes the other's case — is confirmed here."))
def test_a_mention_inside_a_dm_is_handled_once(audit_env, slack_bot, monkeypatch):
    bot, registry = slack_bot
    calls = []
    monkeypatch.setattr(bot, "call_ai",
                        lambda m, p=None, images=None: calls.append(1) or "answer")
    payload = dm("<@U0BOT> status?", ts="10.1")
    event(registry, "message", payload)
    event(registry, "app_mention", payload)
    assert len(calls) == 1, f"the agent ran {len(calls)} times for one message"


@pytest.mark.xfail(strict=True, reason=(
    "FINDING I5 (MEDIUM) — the bot answers @mentions from other bots. "
    "handle_mention (bot.py:997) has no bot_id guard, unlike handle_message "
    "(bot.py:965). Any app that posts a message containing this bot's handle "
    "— an alerting webhook, a CI notifier, another assistant — triggers a full "
    "agent loop attributed to user 'unknown', and two such bots mentioning "
    "each other loop until a budget stops them."))
def test_a_mention_from_another_bot_is_ignored(audit_env, slack_bot, monkeypatch):
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


@pytest.mark.xfail(strict=True, reason=(
    "FINDING I6 (MEDIUM) — a DM with a file attached is silently dropped. "
    "handle_message returns early on ANY subtype (bot.py:965), and Slack tags "
    "a message carrying an upload as subtype 'file_share'. bot.py has a "
    "complete file pipeline behind that early return — image vision, PDF text "
    "extraction, code-file reading (bot.py:484-560) — which a DM can therefore "
    "never reach. Blanket-filtering subtypes is right for message_changed and "
    "message_deleted; file_share is not one of those."))
def test_a_dm_with_a_file_is_processed(audit_env, slack_bot, monkeypatch):
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


@pytest.mark.xfail(strict=True, reason=(
    "FINDING I7 (LOW) — the mention-stripping regex (bot.py:1004, "
    "`<@[A-Z0-9]+>`) does not match Slack's `<@U123|name>` label form, so that "
    "spelling reaches the model and is stored in memory as raw markup. Cosmetic "
    "on its own; it also means the 'bare mention → greeting' path misfires for "
    "the labelled form."))
def test_the_labelled_mention_form_is_stripped(audit_env, slack_bot):
    bot, registry = slack_bot
    event(registry, "app_mention", {"channel": "C1", "ts": "23.1", "user": "U_USER",
                                    "text": "<@U0BOT|agent> summarise"})
    stored = memory.get_history("C1:23.1", limit=5)[0]["content"]
    assert "<@" not in stored, f"raw mention markup stored: {stored!r}"


@pytest.mark.xfail(strict=True, reason=(
    "FINDING I8 (LOW) — a slash payload missing channel_id raises KeyError out "
    "of the listener (bot.py:1044 `command['channel_id']`). Bolt catches it and "
    "logs a 500, so the user sees the command silently do nothing. Every "
    "command reads channel_id directly; command.get('channel_id') with a "
    "response_url fallback would degrade instead."))
def test_a_malformed_slash_payload_does_not_raise(audit_env, slack_bot):
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


@pytest.mark.xfail(strict=True, reason=(
    "FINDING I9 (LOW) — outbound posts have no rate-limit backoff. bot.py "
    "builds WebClient(token=...) with default retry handlers, which are "
    "[ConnectionErrorRetryHandler] only — a Slack 429 raises SlackApiError "
    "rather than being retried. runner._notify (runner.py:528) catches it and "
    "logs, so a rate-limited background run's result is dropped with only a "
    "log line. slack_sdk ships RateLimitErrorRetryHandler for exactly this."))
def test_outbound_posts_retry_on_a_slack_429(audit_env):
    from slack_sdk import WebClient
    handlers = [type(h).__name__ for h in WebClient(token="xoxb-x").retry_handlers]
    assert any("RateLimit" in name for name in handlers), handlers


@pytest.mark.xfail(strict=True, reason=(
    "FINDING I10 (LOW) — the error path can raise the same error again. When "
    "say() fails (channel not joined, Slack degraded), handle_message's except "
    "block answers with another say() (bot.py:994) which fails identically, so "
    "the exception escapes the listener. Bolt's default error handler catches "
    "it, so nothing crashes — but the user gets no message at all, and the "
    "traceback is the only record. The apology post needs its own try/except."))
def test_a_failed_post_never_kills_the_listener(audit_env, slack_bot, monkeypatch):
    """say() raising (not_in_channel, Slack down) must not escape the listener."""
    bot, registry = slack_bot
    monkeypatch.setattr(bot, "call_ai", lambda m, p=None, images=None: "answer")

    def failing_say(text=None, **kw):
        raise RuntimeError("slack_api_error: not_in_channel")

    registry[("event", "message")](dm("hi", ts="24.1"), failing_say)
