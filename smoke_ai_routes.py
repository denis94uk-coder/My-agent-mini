"""Run one harmless request through each configured bot provider.

Run on the server: venv/bin/python smoke_ai_routes.py
The output contains provider names, models, and HTTP statuses, never API keys.
"""

import requests

import bot


def error_text(error: Exception) -> str:
    status = getattr(getattr(error, "response", None), "status_code", None)
    return f"HTTP {status}" if status else type(error).__name__


def call(provider: dict) -> str:
    messages = [{"role": "user", "content": "Reply with exactly: OK"}]
    if provider["type"] == "gemini":
        return bot.call_gemini(provider, messages, bot.SYSTEM_PROMPT)
    if provider["type"] == "cohere":
        return bot.call_cohere(provider, messages, bot.SYSTEM_PROMPT)
    return bot.call_openai_compat(provider, messages, bot.SYSTEM_PROMPT)


def main() -> int:
    bot.build_providers()
    if not bot.PROVIDERS:
        print("No providers configured")
        return 1

    failed = 0
    for provider in bot.PROVIDERS:
        try:
            answer = call(provider)
            if not answer.strip():
                raise RuntimeError("empty response")
            print(f"OK  {provider['name']} ({provider['model']})")
        except (requests.RequestException, RuntimeError, KeyError, IndexError) as error:
            failed += 1
            print(f"FAIL {provider['name']} ({provider['model']}): {error_text(error)}")
    return failed > 0


if __name__ == "__main__":
    raise SystemExit(main())
