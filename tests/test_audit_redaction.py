"""
Phase 1.4 — secret redaction and leak channels.

Invariants under test:
  • Credential shapes emitted by run_shell / run_python never reach the model.
  • Secrets do not escape through the other paths out of the process: the
    durable run_events transcript, the critic's prompt, /costs, or an
    exception message.

The shape denylist is the only control, so the interesting question is not
"does it catch ghp_…" (it does) but where its boundary is. These tests map it.

`xfail(strict=True)` marks a CONFIRMED DEFECT — see it fail with
`pytest --runxfail`.
"""

import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_support import ScriptedAI, audit_env, run_id_of, tool_call  # noqa: F401

import critic
import governor
import runner
import tools

GITHUB = "ghp_" + "A" * 36
TOKENS = {
    "github_classic": GITHUB,
    "github_pat": "github_pat_" + "B" * 30,
    "slack_bot": "xoxb-1234567890-abcdefghijklm",
    "openai": "sk-" + "C" * 32,
    "google": "AIza" + "D" * 35,
    "merge": "mg_" + "E" * 24,
}


@pytest.mark.parametrize("name,secret", sorted(TOKENS.items()))
def test_bare_tokens_are_redacted(audit_env, name, secret):
    out = tools._redact_shell_output(f"$ echo $TOKEN\n{secret}\n")
    assert secret not in out, f"{name} passed through verbatim"


@pytest.mark.parametrize("wrapper", [
    '{{"token": "{s}"}}',
    "Authorization: Bearer {s}",
    "https://x:{s}@github.com/o/r.git",
    "TOKEN={s}",
])
def test_tokens_embedded_in_structure_are_still_redacted(audit_env, wrapper):
    secret = GITHUB
    assert secret not in tools._redact_shell_output(wrapper.format(s=secret))


def test_env_style_keyword_lines_are_redacted(audit_env):
    out = tools._redact_shell_output("SLACK_BOT_TOKEN=xoxb-not-a-real-one\n"
                                     "MY_API_KEY=abcdefabcdef\n")
    assert "xoxb-not-a-real-one" not in out
    assert "abcdefabcdef" not in out


# ── the boundary: any transformation defeats a shape denylist ──

MANGLED = {
    "split_across_lines": GITHUB[:20] + "\n" + GITHUB[20:],
    "base64": base64.b64encode(GITHUB.encode()).decode(),
    "hex": GITHUB.encode().hex(),
    "reversed": GITHUB[::-1],
    "spaced": " ".join(GITHUB[i:i + 8] for i in range(0, len(GITHUB), 8)),
}


def _recover(form: str, text: str) -> str:
    """Undo the transformation — if the secret comes back, it leaked."""
    if form == "split_across_lines":
        return text.replace("\n", "")
    if form == "reversed":
        return text[::-1]
    if form == "spaced":
        return text.replace(" ", "").replace("\n", "")
    if form == "hex":
        out = []
        for token in text.split():
            try:
                out.append(bytes.fromhex(token).decode("utf-8", "replace"))
            except ValueError:
                continue
        return " ".join(out)
    out = []
    for token in text.split():
        try:
            out.append(base64.b64decode(token + "==").decode("utf-8", "replace"))
        except Exception:
            continue
    return " ".join(out)


@pytest.mark.parametrize("form", sorted(MANGLED))
def test_transformed_tokens_are_redacted(audit_env, monkeypatch, form):
    """FIXED (was FINDING G1): shape-matching alone could not survive an
    encoding, so redaction now also knows the values this process actually
    holds and strips them in any form."""
    monkeypatch.setenv("GITHUB_TOKEN", GITHUB)
    out = tools._redact_shell_output(f"$ echo $GITHUB_TOKEN | {form}\n{MANGLED[form]}\n")
    assert GITHUB not in _recover(form, out), f"{form} form reached the model intact"


@pytest.mark.parametrize("form", sorted(MANGLED))
def test_a_foreign_token_is_redacted_when_encoded(audit_env, monkeypatch, form):
    """FIXED (was FINDING G1b): a credential that is NOT ours — read out of a
    file, or pasted into a repo we cloned — is now decoded, un-reversed and
    un-wrapped before the shape patterns run, so the transformation that used
    to hide it no longer does."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    out = tools._redact_shell_output(f"$ cat stolen.txt\n{MANGLED[form]}\n")
    assert GITHUB not in _recover(form, out)


def test_an_encoding_we_do_not_model_still_escapes(audit_env, monkeypatch):
    """ACCEPTED LIMIT. Base64, hex, reversal and whitespace-splitting are
    modelled because they are what a shell one-liner reaches for. A cipher, a
    compressor or a custom alphabet is not, and cannot be — redacting output
    can never be complete. What stands behind it: the tools that read arbitrary
    bytes (run_shell, run_python) are owner-only, so reaching this requires the
    owner's own session, not a stranger's."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    rot13 = GITHUB.translate(str.maketrans(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "nopqrstuvwxyzabcdefghijklmNOPQRSTUVWXYZABCDEFGHIJKLM"))
    out = tools._redact_shell_output(f"$ cat stolen.txt | tr A-Za-z N-ZA-Mn-za-m\n{rot13}\n")
    assert rot13 in out, "if this now redacts, tighten the claim in the report"


@pytest.mark.parametrize("tool_name,kwargs", [
    ("read_file", {"filename": "leak.env"}),
])
def test_file_reading_tools_redact_secrets(audit_env, monkeypatch, tool_name, kwargs):
    """FIXED (was FINDING G2): redaction moved to run_tool, so it covers every
    tool's output rather than the three that remembered to call it."""
    monkeypatch.setattr(tools, "WORKSPACE", str(audit_env))
    (audit_env / "leak.env").write_text(f"GITHUB_TOKEN={GITHUB}\n")
    assert GITHUB not in tools.run_tool(tool_name, dict(kwargs))


# ── other ways out of the process ──

def test_shell_secrets_do_not_reach_the_durable_run_transcript(audit_env, monkeypatch):
    """run_events is written to disk and replayed on resume — a secret there
    outlives the process."""
    monkeypatch.setattr(runner, "_CALL_AI",
                        ScriptedAI(tool_call("run_shell", command="echo $GITHUB_TOKEN"), "done"))
    monkeypatch.setitem(tools.TOOLS["run_shell"], "func",
                        lambda command: tools._redact_shell_output(GITHUB))
    row = runner.enqueue_run("print the token", owner_user_id="U_OWNER", unattended=True)
    runner.execute_run(runner.get_run(run_id_of(row)))
    dumped = "\n".join(e["content"] for e in runner.get_events(run_id_of(row)))
    assert GITHUB not in dumped


def test_the_critic_prompt_carries_only_already_redacted_results(audit_env):
    """The critic gets the tool transcript, so redaction has to happen before
    the result is recorded — not in the Slack formatter."""
    seen = {}

    def spy(messages, prompt=None):
        seen["prompt"] = messages[0]["content"]
        return "VERDICT: ACCEPT"

    steps = [{"tool": "run_shell", "result": tools._redact_shell_output(GITHUB)}]
    critic.review("print it", steps, "done", spy)
    assert GITHUB not in seen["prompt"]


def test_cost_report_contains_no_credentials(audit_env):
    governor.record_ai_call("Gemini", input_chars=10, output_chars=10)
    governor.record_ai_call("Merge Gateway", input_chars=10, output_chars=10)
    report = governor.format_usage()
    assert "key=" not in report and "AIza" not in report and "mg_" not in report


def test_provider_error_never_carries_the_api_key(audit_env, monkeypatch, tmp_path):
    """FIXED (was FINDING G3): the key travels as the x-goog-api-key header, and
    every provider error is scrubbed before it reaches Slack or the log."""
    slack_bolt = pytest.importorskip("slack_bolt")
    import requests

    key = "AIzaSyREAL_LOOKING_KEY_FOR_THE_AUDIT_99"
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake-for-tests")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-fake-for-tests")
    monkeypatch.setenv("POLLINATIONS_ENABLED", "false")

    class FakeApp:  # never touches the network
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, _):
            return lambda *a, **k: (lambda fn: fn)

    monkeypatch.setattr(slack_bolt, "App", FakeApp)
    for name in [m for m in list(sys.modules) if m == "bot"]:
        del sys.modules[name]
    import bot

    def fake_post(url, **kw):
        resp = requests.models.Response()
        resp.status_code = 400
        resp.url = url
        resp.reason = "Bad Request"
        resp._content = b'{"error": {"message": "API key not valid"}}'
        return resp

    monkeypatch.setattr(bot.http_requests, "post", fake_post)
    monkeypatch.setattr(bot, "PROVIDERS", [{
        "name": "Gemini", "type": "gemini", "api_key": key,
        "model": "gemini-2.5-flash",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    }])
    bot.PROVIDER_HEALTH.clear()

    surfaced = bot.call_ai([{"role": "user", "content": "hi"}], "sys")
    assert key not in surfaced, "the API key was returned to the Slack channel"
