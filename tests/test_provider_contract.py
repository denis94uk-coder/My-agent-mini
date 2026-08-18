"""Provider request contracts that must match the live router."""

import sys
from pathlib import Path

import pytest


class _Response:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


@pytest.fixture
def bot_module(monkeypatch, tmp_path):
    import slack_bolt

    class FakeApp:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, _name):
            return lambda *args, **kwargs: (lambda func: func)

    monkeypatch.setattr(slack_bolt, "App", FakeApp)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
    monkeypatch.setenv("POLLINATIONS_ENABLED", "false")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    sys.modules.pop("bot", None)
    import bot
    yield bot
    sys.modules.pop("bot", None)


def test_current_default_models_are_used(bot_module, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    bot_module.build_providers()
    models = {provider["name"]: provider["model"] for provider in bot_module.PROVIDERS}
    assert models["Gemini"] == "gemini-3.6-flash"
    assert models["Groq"] == "openai/gpt-oss-120b"


def test_gemini_uses_a_header_not_a_key_in_the_url(bot_module, monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        return _Response({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    monkeypatch.setattr(bot_module.http_requests, "post", fake_post)
    bot_module.call_gemini({
        "api_key": "secret", "model": "gemini-3.6-flash",
        "url": "https://example.test/models/{model}:generateContent",
    }, [{"role": "user", "content": "hi"}], "system")

    assert "secret" not in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "secret"


def test_gpt_oss_uses_low_reasoning_effort(bot_module, monkeypatch):
    captured = {}

    def fake_post(_url, **kwargs):
        captured["payload"] = kwargs["json"]
        return _Response({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(bot_module.http_requests, "post", fake_post)
    bot_module.call_openai_compat({
        "name": "Groq", "api_key": "secret", "model": "openai/gpt-oss-120b",
        "url": "https://example.test/chat/completions",
    }, [{"role": "user", "content": "hi"}], "system")

    assert captured["payload"]["max_tokens"] == 512
    assert captured["payload"]["reasoning_effort"] == "low"


def test_pdf_file_share_reaches_message_processor(bot_module, monkeypatch):
    seen = {}

    def capture(*_args, **kwargs):
        seen["files"] = kwargs["files"]

    monkeypatch.setattr(bot_module, "process_message", capture)
    bot_module.handle_message({
        "channel": "D1",
        "channel_type": "im",
        "files": [{"name": "report.pdf", "mimetype": "application/pdf"}],
        "subtype": "file_share",
        "text": "",
        "ts": "1.0",
        "user": "U1",
    }, lambda **_message: None)

    assert seen["files"][0]["name"] == "report.pdf"
