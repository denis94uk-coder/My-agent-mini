"""
My-Agent-Mini — API Key Tester
Tests all 10 AI providers to verify your API keys are working.
Run: python test_providers.py
"""

import os
import time
import json
import requests

# ── Load .env file if present ──
from pathlib import Path
env_file = Path(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

# ── Provider definitions ──
PROVIDERS = []

def add_provider(name, env_var, test_fn):
    key = os.getenv(env_var, "").strip()
    if key:
        PROVIDERS.append({"name": name, "env_var": env_var, "key": key, "test": test_fn})

# 1. Gemini
def test_gemini(key):
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {"contents": [{"parts": [{"text": "Say hello in one word."}]}],
               "generationConfig": {"maxOutputTokens": 20}}
    r = requests.post(url, headers={"x-goog-api-key": key}, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"][:80]

# Generic OpenAI-compatible test
def make_openai_test(url, model):
    def test_fn(key):
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        payload = {"model": model, "messages": [{"role": "user", "content": "Say hello in one word."}],
                   "max_tokens": 20, "temperature": 0}
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"].get("content")
        if not content:
            raise RuntimeError("provider returned empty content")
        return content[:80]
    return test_fn

# Cohere test
def test_cohere(key):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    payload = {"model": "command-r-plus", "messages": [{"role": "user", "content": "Say hello in one word."}],
               "max_tokens": 20}
    r = requests.post("https://api.cohere.com/v2/chat", headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"][0]["text"][:80]

# Register all providers
add_provider("Gemini",      "GEMINI_API_KEY",      test_gemini)
add_provider("Groq",        "GROQ_API_KEY",        make_openai_test("https://api.groq.com/openai/v1/chat/completions", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")))
merge_key = os.getenv("MERGE_GATEWAY_API_KEY") or os.getenv("MERGE_API_KEY")
if merge_key:
    PROVIDERS.append({
        "name": "Merge Gateway", "env_var": "MERGE_GATEWAY_API_KEY",
        "key": merge_key,
        "test": make_openai_test(
            "https://api-gateway.merge.dev/v1/openai/chat/completions",
            os.getenv("MERGE_GATEWAY_MODEL", os.getenv("MERGE_MODEL", "openai/gpt-4o-mini")),
        ),
    })
add_provider("Grok (xAI)",  "XAI_API_KEY",         make_openai_test("https://api.x.ai/v1/chat/completions", "grok-3-mini-fast"))
add_provider("Cerebras",    "CEREBRAS_API_KEY",     make_openai_test("https://api.cerebras.ai/v1/chat/completions", "llama-3.3-70b"))
add_provider("SambaNova",   "SAMBANOVA_API_KEY",    make_openai_test("https://api.sambanova.ai/v1/chat/completions", "Meta-Llama-3.3-70B-Instruct"))
add_provider("Together AI", "TOGETHER_API_KEY",     make_openai_test("https://api.together.xyz/v1/chat/completions", "meta-llama/Llama-3.3-70B-Instruct-Turbo"))
add_provider("Mistral",     "MISTRAL_API_KEY",      make_openai_test("https://api.mistral.ai/v1/chat/completions", "mistral-small-latest"))
add_provider("Cohere",      "COHERE_API_KEY",       test_cohere)
add_provider("OpenRouter",  "OPENROUTER_API_KEY",   make_openai_test("https://openrouter.ai/api/v1/chat/completions", "google/gemini-2.0-flash-exp:free"))
add_provider("HuggingFace", "HF_API_KEY",           make_openai_test("https://api-inference.huggingface.co/v1/chat/completions", "meta-llama/Llama-3.3-70B-Instruct"))
add_provider("NVIDIA",     "NVIDIA_API_KEY",    make_openai_test("https://integrate.api.nvidia.com/v1/chat/completions", "nvidia/nemotron-3-ultra-550b-a55b"))

# ── Run tests ──
ALL_KEYS = ["GEMINI_API_KEY", "GROQ_API_KEY", "XAI_API_KEY", "CEREBRAS_API_KEY",
            "SAMBANOVA_API_KEY", "TOGETHER_API_KEY", "MISTRAL_API_KEY",
            "COHERE_API_KEY", "OPENROUTER_API_KEY", "HF_API_KEY",
            "NVIDIA_API_KEY"]

def main():
    print("=" * 60)
    print("  🧪 My-Agent-Mini — API Key Tester")
    print("=" * 60)
    print()

    # Show which keys are set vs missing
    missing = [k for k in ALL_KEYS if not os.getenv(k, "").strip()]
    if missing:
        print(f"⚠️  Missing keys ({len(missing)}):")
        for k in missing:
            print(f"   ❌ {k}")
        print()

    if not PROVIDERS:
        print("❌ No API keys found! Add them to your .env file.")
        return

    print(f"🔍 Testing {len(PROVIDERS)} providers...\n")

    passed = 0
    failed = 0
    results = []

    for p in PROVIDERS:
        name = p["name"]
        print(f"  Testing {name}...", end=" ", flush=True)
        try:
            start = time.time()
            reply = p["test"](p["key"])
            elapsed = round(time.time() - start, 1)
            print(f"✅ OK ({elapsed}s) → \"{reply}\"")
            results.append({"name": name, "status": "✅", "time": f"{elapsed}s"})
            passed += 1
        except Exception as e:
            error = str(e)[:120]
            print(f"❌ FAILED → {error}")
            results.append({"name": name, "status": "❌", "error": error})
            failed += 1

        # Small delay to avoid rate limits
        time.sleep(0.5)

    # Summary
    print()
    print("=" * 60)
    print(f"  📊 Results: {passed} passed ✅  |  {failed} failed ❌  |  {len(missing)} not configured")
    print("=" * 60)

    if failed == 0 and not missing:
        print("\n  🎉 All providers working perfectly!")
    elif failed > 0:
        print("\n  ⚠️  Check the failed providers above.")
        print("     Common fixes:")
        print("     • Wrong API key → regenerate it")
        print("     • Free tier expired → check the provider dashboard")
        print("     • Rate limited → wait a minute and try again")

if __name__ == "__main__":
    main()
