"""
My-Agent-Mini — Lightweight Hybrid AI Slack Bot
Runs on Oracle Cloud E2.1.Micro (1 OCPU / 1 GB RAM) or any small VPS.
Calls free AI APIs directly — no LiteLLM, no Docker, no database needed.
Hybrid failover: if one provider is down or out of credits, tries the next.
"""

import os
import re
import json
import time
import logging
import traceback
from pathlib import Path

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("my-agent-mini")

# ── Slack Config ──
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]      # xoxb-...
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]       # xapp-...
BOT_NAME = os.getenv("BOT_NAME", "My Agent")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT",
    f"You are {BOT_NAME}, a helpful AI assistant powered by multiple free AI providers. "
    "You help with coding, research, analysis, writing, and any question. "
    "Be concise but thorough. Use markdown formatting suitable for Slack."
)

# ── AI Provider Config ──
# Each provider is tried in order. If one fails, the next is tried.
# Set the API key env vars for the providers you want to use.
# You only need ONE working provider, but more = better reliability.

PROVIDERS = []  # Built at startup from available env vars


def build_providers():
    """Build list of available AI providers from environment variables."""
    global PROVIDERS
    PROVIDERS = []

    # 1. Google Gemini (most generous free tier: 15 RPM, 1M tokens/day)
    if os.getenv("GEMINI_API_KEY"):
        PROVIDERS.append({
            "name": "Gemini",
            "type": "gemini",
            "api_key": os.environ["GEMINI_API_KEY"],
            "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        })

    # 2. Groq (very fast, free: 30 RPM)
    if os.getenv("GROQ_API_KEY"):
        PROVIDERS.append({
            "name": "Groq",
            "type": "openai_compat",
            "api_key": os.environ["GROQ_API_KEY"],
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "url": "https://api.groq.com/openai/v1/chat/completions",
        })

    # 3. xAI / Grok (free tier: 60 RPM)
    if os.getenv("XAI_API_KEY"):
        PROVIDERS.append({
            "name": "Grok",
            "type": "openai_compat",
            "api_key": os.environ["XAI_API_KEY"],
            "model": os.getenv("XAI_MODEL", "grok-3-mini-fast"),
            "url": "https://api.x.ai/v1/chat/completions",
        })

    # 4. Cerebras (ultra-fast inference, free tier)
    if os.getenv("CEREBRAS_API_KEY"):
        PROVIDERS.append({
            "name": "Cerebras",
            "type": "openai_compat",
            "api_key": os.environ["CEREBRAS_API_KEY"],
            "model": os.getenv("CEREBRAS_MODEL", "llama-3.3-70b"),
            "url": "https://api.cerebras.ai/v1/chat/completions",
        })

    # 5. SambaNova (free fast inference)
    if os.getenv("SAMBANOVA_API_KEY"):
        PROVIDERS.append({
            "name": "SambaNova",
            "type": "openai_compat",
            "api_key": os.environ["SAMBANOVA_API_KEY"],
            "model": os.getenv("SAMBANOVA_MODEL", "Meta-Llama-3.3-70B-Instruct"),
            "url": "https://api.sambanova.ai/v1/chat/completions",
        })

    # 6. Together AI (free tier: $5 credit)
    if os.getenv("TOGETHER_API_KEY"):
        PROVIDERS.append({
            "name": "Together",
            "type": "openai_compat",
            "api_key": os.environ["TOGETHER_API_KEY"],
            "model": os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
            "url": "https://api.together.xyz/v1/chat/completions",
        })

    # 7. Mistral (free tier)
    if os.getenv("MISTRAL_API_KEY"):
        PROVIDERS.append({
            "name": "Mistral",
            "type": "openai_compat",
            "api_key": os.environ["MISTRAL_API_KEY"],
            "model": os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
            "url": "https://api.mistral.ai/v1/chat/completions",
        })

    # 8. Cohere (free tier: 20 RPM)
    if os.getenv("COHERE_API_KEY"):
        PROVIDERS.append({
            "name": "Cohere",
            "type": "cohere",
            "api_key": os.environ["COHERE_API_KEY"],
            "model": os.getenv("COHERE_MODEL", "command-r-plus"),
            "url": "https://api.cohere.com/v2/chat",
        })

    # 9. OpenRouter (free models available)
    if os.getenv("OPENROUTER_API_KEY"):
        PROVIDERS.append({
            "name": "OpenRouter",
            "type": "openai_compat",
            "api_key": os.environ["OPENROUTER_API_KEY"],
            "model": os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free"),
            "url": "https://openrouter.ai/api/v1/chat/completions",
        })

    # 10. HuggingFace Inference API (free tier)
    if os.getenv("HF_API_KEY"):
        PROVIDERS.append({
            "name": "HuggingFace",
            "type": "openai_compat",
            "api_key": os.environ["HF_API_KEY"],
            "model": os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
            "url": "https://api-inference.huggingface.co/v1/chat/completions",
        })

    logger.info(f"🧠 Loaded {len(PROVIDERS)} AI providers: {[p['name'] for p in PROVIDERS]}")


# ── AI Calling Logic ──
import requests as http_requests  # renamed to avoid clash with slack events


def call_gemini(provider: dict, messages: list[dict]) -> str:
    """Call Google Gemini API."""
    url = provider["url"].format(model=provider["model"])
    url += f"?key={provider['api_key']}"

    # Convert messages to Gemini format
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })

    # Add system instruction separately
    payload = {
        "contents": contents,
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "generationConfig": {
            "maxOutputTokens": 3000,
            "temperature": 0.7,
        }
    }

    resp = http_requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def call_openai_compat(provider: dict, messages: list[dict]) -> str:
    """Call any OpenAI-compatible API (Groq, xAI, Cerebras, Together, etc.)."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {provider['api_key']}",
    }

    payload = {
        "model": provider["model"],
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "max_tokens": 3000,
        "temperature": 0.7,
    }

    resp = http_requests.post(provider["url"], headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_cohere(provider: dict, messages: list[dict]) -> str:
    """Call Cohere API."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {provider['api_key']}",
    }

    # Convert to Cohere format
    cohere_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in messages:
        role = msg["role"]
        if role == "user":
            cohere_messages.append({"role": "user", "content": msg["content"]})
        else:
            cohere_messages.append({"role": "assistant", "content": msg["content"]})

    payload = {
        "model": provider["model"],
        "messages": cohere_messages,
        "max_tokens": 3000,
        "temperature": 0.7,
    }

    resp = http_requests.post(provider["url"], headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"][0]["text"]


def call_ai(messages: list[dict]) -> str:
    """Try each AI provider in order until one succeeds."""
    if not PROVIDERS:
        return "❌ No AI providers configured! Add at least one API key to your `.env` file."

    errors = []
    for provider in PROVIDERS:
        try:
            logger.info(f"🔄 Trying {provider['name']} ({provider['model']})...")
            start = time.time()

            if provider["type"] == "gemini":
                result = call_gemini(provider, messages)
            elif provider["type"] == "cohere":
                result = call_cohere(provider, messages)
            else:
                result = call_openai_compat(provider, messages)

            elapsed = round(time.time() - start, 1)
            logger.info(f"✅ {provider['name']} responded in {elapsed}s")
            return result

        except Exception as e:
            error_msg = str(e)[:200]
            logger.warning(f"⚠️ {provider['name']} failed: {error_msg}")
            errors.append(f"• {provider['name']}: {error_msg}")
            continue

    # All providers failed
    error_list = "\n".join(errors)
    return f"❌ All AI providers failed:\n{error_list}\n\nPlease check your API keys or try again later."


# ── Conversation Memory (in-memory, lightweight) ──
conversations: dict[str, list[dict]] = {}


def get_conv_key(channel: str, thread_ts: str | None) -> str:
    return f"{channel}:{thread_ts or 'main'}"


def get_history(key: str) -> list[dict]:
    if key not in conversations:
        conversations[key] = []
    return conversations[key]


def add_to_history(key: str, role: str, content: str):
    history = get_history(key)
    history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY * 2:
        conversations[key] = history[-(MAX_HISTORY * 2):]


def truncate_for_slack(text: str, limit: int = 3900) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n_(truncated — reply to continue)_"


# ── Slack App ──
app = App(token=SLACK_BOT_TOKEN)


@app.event("message")
def handle_message(event, say):
    """Handle DMs."""
    if event.get("bot_id") or event.get("subtype"):
        return

    channel = event["channel"]
    thread_ts = event.get("thread_ts") or event.get("ts")
    user_text = event.get("text", "").strip()
    channel_type = event.get("channel_type", "")

    if not user_text or channel_type not in ("im", "mpim"):
        return

    conv_key = get_conv_key(channel, thread_ts)
    add_to_history(conv_key, "user", user_text)

    logger.info(f"💬 DM: {user_text[:80]}...")
    response = call_ai(get_history(conv_key))
    add_to_history(conv_key, "assistant", response)

    say(text=truncate_for_slack(response), thread_ts=thread_ts)


@app.event("app_mention")
def handle_mention(event, say):
    """Handle @mentions in channels."""
    channel = event["channel"]
    thread_ts = event.get("thread_ts") or event["ts"]
    user_text = re.sub(r"<@[A-Z0-9]+>\s*", "", event.get("text", "")).strip()

    if not user_text:
        say(text=f"Hey! I'm {BOT_NAME} 👋 Ask me anything!", thread_ts=thread_ts)
        return

    conv_key = get_conv_key(channel, thread_ts)
    add_to_history(conv_key, "user", user_text)

    logger.info(f"📢 Mention: {user_text[:80]}...")
    response = call_ai(get_history(conv_key))
    add_to_history(conv_key, "assistant", response)

    say(text=truncate_for_slack(response), thread_ts=thread_ts)


@app.command("/ask")
def handle_ask(ack, command, say):
    """/ask <question> — Quick one-shot question."""
    ack()
    question = command.get("text", "").strip()
    if not question:
        say("Usage: `/ask <your question>`", channel=command["channel_id"])
        return

    logger.info(f"❓ /ask: {question[:80]}...")
    response = call_ai([{"role": "user", "content": question}])
    say(
        text=f"*Q:* {question}\n\n{truncate_for_slack(response)}",
        channel=command["channel_id"],
    )


@app.command("/clear")
def handle_clear(ack, command, say):
    """/clear — Reset conversation memory."""
    ack()
    channel = command["channel_id"]
    keys_to_delete = [k for k in conversations if k.startswith(f"{channel}:")]
    for k in keys_to_delete:
        del conversations[k]
    say(text="🧹 Conversation memory cleared!", channel=channel)


@app.command("/providers")
def handle_providers(ack, command, say):
    """/providers — Show active AI providers."""
    ack()
    if not PROVIDERS:
        say(text="❌ No providers configured.", channel=command["channel_id"])
        return

    lines = [f"🧠 *Active AI Providers ({len(PROVIDERS)}):*\n"]
    for i, p in enumerate(PROVIDERS, 1):
        lines.append(f"{i}. *{p['name']}* — `{p['model']}`")
    lines.append(f"\n_Providers are tried in order. If one fails, the next is used automatically._")
    say(text="\n".join(lines), channel=command["channel_id"])


# ── Start ──
if __name__ == "__main__":
    build_providers()

    if not PROVIDERS:
        logger.warning("⚠️  No AI providers configured! Add API keys to .env")
        logger.warning("   The bot will start but won't be able to answer questions.")

    logger.info(f"🤖 {BOT_NAME} starting (Socket Mode)...")
    logger.info(f"   Providers: {len(PROVIDERS)}")
    logger.info(f"   History: {MAX_HISTORY} messages per thread")

    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
