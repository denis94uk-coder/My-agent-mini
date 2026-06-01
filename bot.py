"""
My-Agent-Mini v2 — AI Agent Slack Bot
Runs on Google Cloud e2-micro (1 GB RAM) or any small VPS.

Upgrades from v1:
  ✅ Persistent memory (SQLite) — survives restarts
  ✅ ReAct agent loop — multi-step task execution
  ✅ Web search (DuckDuckGo) — real-time information
  ✅ URL fetching — read any webpage
  ✅ Python execution — run code in a sandbox
  ✅ Memory search — recall past conversations
  ✅ User facts — remembers things about you

Still lightweight: ~50 MB RAM, no Docker needed.
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

import memory
import tools
import agent

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
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT",
    f"You are {BOT_NAME}, a powerful AI assistant running as a Slack bot. "
    "You can search the web, read URLs, execute Python code, and remember things. "
    "You help with coding, research, analysis, writing, math, and any question. "
    "Be concise but thorough. Use markdown formatting suitable for Slack. "
    "When using tools, explain what you're doing so the user can follow along."
)

# ── AI Provider Config ──
PROVIDERS = []

import requests as http_requests


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

def call_gemini(provider: dict, messages: list[dict], system_prompt: str) -> str:
    """Call Google Gemini API."""
    url = provider["url"].format(model=provider["model"])
    url += f"?key={provider['api_key']}"

    contents = []
    for msg in messages:
        if msg["role"] == "system":
            continue
        role = "user" if msg["role"] == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })

    payload = {
        "contents": contents,
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "maxOutputTokens": 4000,
            "temperature": 0.7,
        }
    }

    resp = http_requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def call_openai_compat(provider: dict, messages: list[dict], system_prompt: str) -> str:
    """Call any OpenAI-compatible API (Groq, xAI, Cerebras, Together, etc.)."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {provider['api_key']}",
    }

    payload = {
        "model": provider["model"],
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "max_tokens": 4000,
        "temperature": 0.7,
    }

    resp = http_requests.post(provider["url"], headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_cohere(provider: dict, messages: list[dict], system_prompt: str) -> str:
    """Call Cohere API."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {provider['api_key']}",
    }

    cohere_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        role = msg["role"] if msg["role"] in ("user", "assistant") else "user"
        cohere_messages.append({"role": role, "content": msg["content"]})

    payload = {
        "model": provider["model"],
        "messages": cohere_messages,
        "max_tokens": 4000,
        "temperature": 0.7,
    }

    resp = http_requests.post(provider["url"], headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"][0]["text"]


def call_ai(messages: list[dict], system_prompt: str = None) -> str:
    """Try each AI provider in order until one succeeds."""
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    if not PROVIDERS:
        return "❌ No AI providers configured! Add at least one API key to your `.env` file."

    errors = []
    for provider in PROVIDERS:
        try:
            logger.info(f"🔄 Trying {provider['name']} ({provider['model']})...")
            start = time.time()

            if provider["type"] == "gemini":
                result = call_gemini(provider, messages, system_prompt)
            elif provider["type"] == "cohere":
                result = call_cohere(provider, messages, system_prompt)
            else:
                result = call_openai_compat(provider, messages, system_prompt)

            elapsed = round(time.time() - start, 1)
            logger.info(f"✅ {provider['name']} responded in {elapsed}s")
            return result

        except Exception as e:
            error_msg = str(e)[:200]
            logger.warning(f"⚠️ {provider['name']} failed: {error_msg}")
            errors.append(f"• {provider['name']}: {error_msg}")
            continue

    error_list = "\n".join(errors)
    return f"❌ All AI providers failed:\n{error_list}\n\nPlease check your API keys or try again later."


def truncate_for_slack(text: str, limit: int = 3900) -> str:
    """Truncate text to fit in a single Slack message."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n_(truncated — reply to continue)_"


# ── Slack App ──
slack_app = App(token=SLACK_BOT_TOKEN)


def get_conv_key(channel: str, thread_ts: str | None) -> str:
    return f"{channel}:{thread_ts or 'main'}"


def process_message(user_text: str, channel: str, thread_ts: str, user_id: str, say):
    """Process a message through the agent loop."""
    conv_key = get_conv_key(channel, thread_ts)

    # Store in persistent memory
    memory.add_message(conv_key, "user", user_text)

    # Get conversation history
    history = memory.get_history(conv_key, limit=MAX_HISTORY)

    # Run through agent loop (handles tool calls automatically)
    response = agent.run_agent_loop(
        messages=history,
        call_ai_fn=call_ai,
        system_prompt=SYSTEM_PROMPT,
        user_id=user_id,
    )

    # Clean up any leftover tool call syntax from the response
    response = agent.extract_final_text(response)

    # Store response in memory
    memory.add_message(conv_key, "assistant", response)

    # Send to Slack
    say(text=truncate_for_slack(response), thread_ts=thread_ts)


@slack_app.event("message")
def handle_message(event, say):
    """Handle DMs."""
    if event.get("bot_id") or event.get("subtype"):
        return

    channel = event["channel"]
    thread_ts = event.get("thread_ts") or event.get("ts")
    user_text = event.get("text", "").strip()
    user_id = event.get("user", "unknown")
    channel_type = event.get("channel_type", "")

    if not user_text or channel_type not in ("im", "mpim"):
        return

    logger.info(f"💬 DM from {user_id}: {user_text[:80]}...")

    try:
        process_message(user_text, channel, thread_ts, user_id, say)
    except Exception as e:
        logger.error(f"❌ Error processing message: {traceback.format_exc()}")
        say(text=f"Sorry, I hit an error: {str(e)[:200]}", thread_ts=thread_ts)


@slack_app.event("app_mention")
def handle_mention(event, say):
    """Handle @mentions in channels."""
    channel = event["channel"]
    thread_ts = event.get("thread_ts") or event["ts"]
    user_id = event.get("user", "unknown")
    user_text = re.sub(r"<@[A-Z0-9]+>\s*", "", event.get("text", "")).strip()

    if not user_text:
        say(text=f"Hey! I'm {BOT_NAME} 👋 Ask me anything — I can search the web, run code, and more!", thread_ts=thread_ts)
        return

    logger.info(f"📢 Mention from {user_id}: {user_text[:80]}...")

    try:
        process_message(user_text, channel, thread_ts, user_id, say)
    except Exception as e:
        logger.error(f"❌ Error processing mention: {traceback.format_exc()}")
        say(text=f"Sorry, I hit an error: {str(e)[:200]}", thread_ts=thread_ts)


# ── Slash Commands ──

@slack_app.command("/ask")
def handle_ask(ack, command, say):
    """/ask <question> — Quick one-shot question (no memory)."""
    ack()
    question = command.get("text", "").strip()
    if not question:
        say("Usage: `/ask <your question>`", channel=command["channel_id"])
        return

    logger.info(f"❓ /ask: {question[:80]}...")
    response = agent.run_agent_loop(
        messages=[{"role": "user", "content": question}],
        call_ai_fn=call_ai,
        system_prompt=SYSTEM_PROMPT,
        user_id=command.get("user_id", "unknown"),
    )
    response = agent.extract_final_text(response)
    say(
        text=f"*Q:* {question}\n\n{truncate_for_slack(response)}",
        channel=command["channel_id"],
    )


@slack_app.command("/search")
def handle_search(ack, command, say):
    """/search <query> — Search the web."""
    ack()
    query = command.get("text", "").strip()
    if not query:
        say("Usage: `/search <what to search for>`", channel=command["channel_id"])
        return

    logger.info(f"🔍 /search: {query[:80]}...")
    result = tools.run_tool("web_search", {"query": query})
    say(
        text=f"🔍 *Search:* {query}\n\n{truncate_for_slack(result)}",
        channel=command["channel_id"],
    )


@slack_app.command("/clear")
def handle_clear(ack, command, say):
    """/clear — Reset conversation memory for this channel."""
    ack()
    channel = command["channel_id"]
    # Clear all conversations starting with this channel
    conn = memory.get_db()
    try:
        conn.execute("DELETE FROM conversations WHERE conv_key LIKE ?", (f"{channel}:%",))
        conn.commit()
    finally:
        conn.close()
    say(text="🧹 Conversation memory cleared!", channel=channel)


@slack_app.command("/providers")
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


@slack_app.command("/status")
def handle_status(ack, command, say):
    """/status — Show bot status and memory stats."""
    ack()
    stats = memory.get_stats()
    status_text = (
        f"🤖 *{BOT_NAME} Status*\n\n"
        f"*Providers:* {len(PROVIDERS)} active\n"
        f"*Memory:*\n"
        f"  • {stats['messages']} messages stored\n"
        f"  • {stats['facts']} facts remembered\n"
        f"  • {stats['conversations']} conversations\n"
        f"  • Database size: {stats['db_size_mb']} MB\n\n"
        f"*Tools:* {', '.join(tools.TOOLS.keys())}\n\n"
        f"_Running on e2-micro • v2.0 with agent loop_"
    )
    say(text=status_text, channel=command["channel_id"])


# ── Start ──
if __name__ == "__main__":
    build_providers()

    if not PROVIDERS:
        logger.warning("⚠️  No AI providers configured! Add API keys to .env")
        logger.warning("   The bot will start but won't be able to answer questions.")

    logger.info(f"🤖 {BOT_NAME} v2.0 starting (Socket Mode)...")
    logger.info(f"   Providers: {len(PROVIDERS)}")
    logger.info(f"   History: {MAX_HISTORY} messages per thread")
    logger.info(f"   Tools: {list(tools.TOOLS.keys())}")
    logger.info(f"   Memory: SQLite at {memory.DB_PATH}")

    handler = SocketModeHandler(slack_app, SLACK_APP_TOKEN)
    handler.start()
