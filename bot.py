"""
My-Agent-Mini v4 — Full Agent: reasoning framework + real execution
Runs on Google Cloud e2-micro (1 GB RAM) or any small VPS.

v4 upgrades:
  ✅ 4-phase reasoning framework (Understand → Plan → Execute → Deliver)
  ✅ Shell access (run_shell) — install packages, git, system tasks
  ✅ Persistent file workspace (write_file, read_file, list_files)
  ✅ 10 tool steps per task (was 5) — completes multi-step work
  ✅ Robust tool-call parsing (handles nested code and braces)

Previous features (v3):
  ✅ Image understanding (Gemini Vision — sees photos, screenshots, diagrams)
  ✅ Document reading (PDF, text, code files shared in Slack)
  ✅ Smarter system prompt (focused, actionable, less generic)
  ✅ Better error messages and user experience
  ✅ File download from Slack (images + documents)
  ✅ Multi-image support (analyze multiple images at once)

Previous features (v2):
  ✅ Persistent memory (SQLite)
  ✅ ReAct agent loop
  ✅ Web search, URL fetching, Python execution
  ✅ Memory search and user facts
"""

import os
import re
import io
import json
import time
import base64
import logging
import mimetypes
import traceback
from pathlib import Path

# ── Load .env file (must happen BEFORE reading any tokens) ──
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    print("⚠️  python-dotenv not installed — run: pip3 install python-dotenv")

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient

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
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "")
if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
    print("❌ Missing Slack tokens!")
    print("   Make sure the .env file exists in this folder and contains:")
    print("   SLACK_BOT_TOKEN=xoxb-...")
    print("   SLACK_APP_TOKEN=xapp-...")
    print("   Check with:  grep SLACK .env")
    raise SystemExit(1)
BOT_NAME = os.getenv("BOT_NAME", "My Agent")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))

# Slack client for file downloads
slack_client = WebClient(token=SLACK_BOT_TOKEN)

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT",
    f"""You are {BOT_NAME}, a sharp AI assistant on Slack.

PERSONALITY:
- Be direct and actionable — skip filler phrases like "Great question!" or "Sure!"
- Give concrete answers, not overviews. If someone asks how to do X, show them.
- Use short paragraphs. Bullet points for lists. Code blocks for code.
- When you don't know something, say so and offer to search the web.
- Remember past conversations and facts about the user.

CAPABILITIES:
- 🔍 Search the web for real-time info
- 🌐 Read any webpage or URL
- 🐍 Run Python code for calculations, data processing
- 💻 Run shell commands on the server (install, git, files, system tasks)
- 📁 Save and read files in a persistent workspace
- 🖼️ Analyze images (screenshots, photos, diagrams, documents)
- 📄 Read uploaded files (PDF, text, code, CSV)
- 🧠 Remember things about users across conversations

You are a doer, not just a talker. When asked to accomplish something,
actually do it with your tools, verify it worked, and report the result.

FORMAT FOR SLACK:
- Use *bold* for emphasis (not **bold**)
- Use `code` for inline code
- Use ```code blocks``` for multi-line code
- Use > for quotes
- Keep responses concise — expand only if asked"""
)

# ── Operating Manual (optional) ──
# If operating_manual.md exists next to bot.py, it is appended to the
# system prompt and governs every response the agent produces.
# Edit the file, restart the bot, and the new rules apply.
MANUAL_PATH = Path(__file__).parent / "operating_manual.md"
OPERATING_MANUAL = ""
try:
    if MANUAL_PATH.exists():
        OPERATING_MANUAL = MANUAL_PATH.read_text(encoding="utf-8").strip()
except Exception as e:
    print(f"⚠️  Could not read operating_manual.md: {e}")

if OPERATING_MANUAL:
    SYSTEM_PROMPT += (
        "\n\n═══════════ OPERATING MANUAL ═══════════\n"
        "The following manual governs every response you produce. "
        "When a rule below conflicts with a request's phrasing, "
        "the rule that protects correctness wins.\n\n"
        + OPERATING_MANUAL
    )

# ── AI Provider Config ──
PROVIDERS = []

import requests as http_requests


def build_providers():
    """Build list of available AI providers from environment variables."""
    global PROVIDERS
    PROVIDERS = []

    if os.getenv("GEMINI_API_KEY"):
        PROVIDERS.append({
            "name": "Gemini",
            "type": "gemini",
            "api_key": os.environ["GEMINI_API_KEY"],
            "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        })

    if os.getenv("GROQ_API_KEY"):
        PROVIDERS.append({
            "name": "Groq",
            "type": "openai_compat",
            "api_key": os.environ["GROQ_API_KEY"],
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "url": "https://api.groq.com/openai/v1/chat/completions",
        })

    if os.getenv("XAI_API_KEY"):
        PROVIDERS.append({
            "name": "Grok",
            "type": "openai_compat",
            "api_key": os.environ["XAI_API_KEY"],
            "model": os.getenv("XAI_MODEL", "grok-3-mini-fast"),
            "url": "https://api.x.ai/v1/chat/completions",
        })

    if os.getenv("CEREBRAS_API_KEY"):
        PROVIDERS.append({
            "name": "Cerebras",
            "type": "openai_compat",
            "api_key": os.environ["CEREBRAS_API_KEY"],
            "model": os.getenv("CEREBRAS_MODEL", "llama-3.3-70b"),
            "url": "https://api.cerebras.ai/v1/chat/completions",
        })

    if os.getenv("SAMBANOVA_API_KEY"):
        PROVIDERS.append({
            "name": "SambaNova",
            "type": "openai_compat",
            "api_key": os.environ["SAMBANOVA_API_KEY"],
            "model": os.getenv("SAMBANOVA_MODEL", "Meta-Llama-3.3-70B-Instruct"),
            "url": "https://api.sambanova.ai/v1/chat/completions",
        })

    if os.getenv("TOGETHER_API_KEY"):
        PROVIDERS.append({
            "name": "Together",
            "type": "openai_compat",
            "api_key": os.environ["TOGETHER_API_KEY"],
            "model": os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
            "url": "https://api.together.xyz/v1/chat/completions",
        })

    if os.getenv("MISTRAL_API_KEY"):
        PROVIDERS.append({
            "name": "Mistral",
            "type": "openai_compat",
            "api_key": os.environ["MISTRAL_API_KEY"],
            "model": os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
            "url": "https://api.mistral.ai/v1/chat/completions",
        })

    if os.getenv("COHERE_API_KEY"):
        PROVIDERS.append({
            "name": "Cohere",
            "type": "cohere",
            "api_key": os.environ["COHERE_API_KEY"],
            "model": os.getenv("COHERE_MODEL", "command-r-plus"),
            "url": "https://api.cohere.com/v2/chat",
        })

    if os.getenv("OPENROUTER_API_KEY"):
        PROVIDERS.append({
            "name": "OpenRouter",
            "type": "openai_compat",
            "api_key": os.environ["OPENROUTER_API_KEY"],
            "model": os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free"),
            "url": "https://openrouter.ai/api/v1/chat/completions",
        })

    if os.getenv("HF_API_KEY"):
        PROVIDERS.append({
            "name": "HuggingFace",
            "type": "openai_compat",
            "api_key": os.environ["HF_API_KEY"],
            "model": os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
            "url": "https://api-inference.huggingface.co/v1/chat/completions",
        })

    # 11. Merge Gateway (OpenAI-compatible proxy — Claude, GPT, etc.)
    if os.getenv("MERGE_API_KEY"):
        PROVIDERS.append({
            "name": "Merge",
            "type": "openai_compat",
            "api_key": os.environ["MERGE_API_KEY"],
            "model": os.getenv("MERGE_MODEL", "anthropic/claude-sonnet-4-20250514"),
            "url": "https://api-gateway.merge.dev/v1/openai/chat/completions",
        })

    # 12. NVIDIA NIM (free 1,000 API calls)
    if os.getenv("NVIDIA_API_KEY"):
        PROVIDERS.append({
            "name": "NVIDIA",
            "type": "openai_compat",
            "api_key": os.environ["NVIDIA_API_KEY"],
            "model": os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
            "url": "https://integrate.api.nvidia.com/v1/chat/completions",
        })

    logger.info(f"🧠 Loaded {len(PROVIDERS)} AI providers: {[p['name'] for p in PROVIDERS]}")


# ── File Handling ──

# Image types that Gemini Vision can process
IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}

# Document types we can extract text from
DOC_TYPES = {
    "text/plain", "text/csv", "text/html", "text/markdown",
    "application/json", "application/xml",
    "application/pdf",
    "application/javascript", "text/x-python", "text/x-java-source",
}

# Extensions we treat as text even if MIME is wrong
TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".xml", ".html", ".css", ".js", ".ts",
    ".py", ".java", ".rb", ".go", ".rs", ".c", ".cpp", ".h", ".sh", ".bash",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env", ".log",
    ".sql", ".r", ".php", ".swift", ".kt", ".scala", ".lua",
}


def download_slack_file(url: str) -> bytes:
    """Download a file from Slack using the bot token."""
    resp = http_requests.get(
        url,
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.content


def extract_text_from_pdf(data: bytes) -> str:
    """Extract text from a PDF file."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        text_parts = []
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text().strip()
            if page_text:
                text_parts.append(f"--- Page {page_num} ---\n{page_text}")
        doc.close()
        return "\n\n".join(text_parts) if text_parts else "(PDF has no extractable text — may be a scanned image)"
    except ImportError:
        # Fallback without PyMuPDF
        return "(PDF reading requires PyMuPDF — install with: pip install PyMuPDF)"
    except Exception as e:
        return f"(Error reading PDF: {str(e)[:200]})"


def is_text_file(filename: str, mimetype: str) -> bool:
    """Check if a file should be treated as text."""
    ext = Path(filename).suffix.lower() if filename else ""
    if ext in TEXT_EXTENSIONS:
        return True
    if mimetype and (mimetype.startswith("text/") or mimetype in DOC_TYPES):
        return True
    return False


def process_slack_files(files: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Process files attached to a Slack message.
    Returns: (images_for_vision, text_descriptions)
    
    images_for_vision: list of {"mime_type": str, "data": base64_str}
    text_descriptions: list of text strings to prepend to the user message
    """
    images = []
    texts = []

    for f in files:
        filename = f.get("name", "unknown")
        mimetype = f.get("mimetype", "")
        file_url = f.get("url_private_download") or f.get("url_private", "")
        file_size = f.get("size", 0)

        if not file_url:
            texts.append(f"📎 {filename} (no download URL available)")
            continue

        # Skip very large files (>10 MB)
        if file_size > 10 * 1024 * 1024:
            texts.append(f"📎 {filename} (skipped — {file_size // 1024 // 1024} MB, too large)")
            continue

        logger.info(f"📥 Processing file: {filename} ({mimetype}, {file_size} bytes)")

        try:
            data = download_slack_file(file_url)

            # Image → send to vision
            if mimetype in IMAGE_TYPES:
                b64 = base64.b64encode(data).decode("utf-8")
                images.append({"mime_type": mimetype, "data": b64})
                texts.append(f"🖼️ [Image: {filename}]")
                logger.info(f"  → Image ready for vision ({len(data)} bytes)")

            # PDF → extract text
            elif mimetype == "application/pdf":
                pdf_text = extract_text_from_pdf(data)
                if len(pdf_text) > 6000:
                    pdf_text = pdf_text[:6000] + "\n\n... (truncated, document is very long)"
                texts.append(f"📄 *Document: {filename}*\n```\n{pdf_text}\n```")
                logger.info(f"  → PDF text extracted ({len(pdf_text)} chars)")

            # Text/code files → read content
            elif is_text_file(filename, mimetype):
                try:
                    text_content = data.decode("utf-8", errors="replace")
                except:
                    text_content = data.decode("latin-1", errors="replace")
                if len(text_content) > 6000:
                    text_content = text_content[:6000] + "\n\n... (truncated)"
                ext = Path(filename).suffix.lstrip(".")
                texts.append(f"📄 *File: {filename}*\n```{ext}\n{text_content}\n```")
                logger.info(f"  → Text file read ({len(text_content)} chars)")

            else:
                texts.append(f"📎 {filename} ({mimetype} — unsupported file type)")

        except Exception as e:
            logger.error(f"  → Failed to process {filename}: {e}")
            texts.append(f"📎 {filename} (failed to download: {str(e)[:100]})")

    return images, texts


# ── AI Calling Logic ──

def call_gemini(provider: dict, messages: list[dict], system_prompt: str, images: list[dict] = None) -> str:
    """Call Google Gemini API with optional vision (images)."""
    url = provider["url"].format(model=provider["model"])
    url += f"?key={provider['api_key']}"

    contents = []
    for msg in messages:
        if msg["role"] == "system":
            continue
        role = "user" if msg["role"] == "user" else "model"

        # Build parts for this message
        parts = []

        # Add text content
        if msg.get("content"):
            parts.append({"text": msg["content"]})

        # Add images to the LAST user message only
        if role == "user" and images and msg == messages[-1]:
            for img in images:
                parts.append({
                    "inline_data": {
                        "mime_type": img["mime_type"],
                        "data": img["data"],
                    }
                })

        if parts:
            contents.append({"role": role, "parts": parts})

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


def call_openai_compat(provider: dict, messages: list[dict], system_prompt: str, images: list[dict] = None) -> str:
    """Call any OpenAI-compatible API."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {provider['api_key']}",
    }

    api_messages = [{"role": "system", "content": system_prompt}]

    for msg in messages:
        # For the last user message, include images if the provider supports it
        if images and msg == messages[-1] and msg["role"] == "user":
            content_parts = [{"type": "text", "text": msg["content"]}]
            for img in images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img['mime_type']};base64,{img['data']}"
                    }
                })
            api_messages.append({"role": "user", "content": content_parts})
        else:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

    payload = {
        "model": provider["model"],
        "messages": api_messages,
        "max_tokens": 4000,
        "temperature": 0.7,
    }

    resp = http_requests.post(provider["url"], headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_cohere(provider: dict, messages: list[dict], system_prompt: str, images: list[dict] = None) -> str:
    """Call Cohere API (no vision support)."""
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


def call_ai(messages: list[dict], system_prompt: str = None, images: list[dict] = None) -> str:
    """Try each AI provider in order until one succeeds."""
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    if not PROVIDERS:
        return "❌ No AI providers configured! Add at least one API key to your `.env` file."

    # If we have images, prefer Gemini (best free vision)
    providers_order = list(PROVIDERS)
    if images:
        # Move Gemini to front if available
        gemini = [p for p in providers_order if p["type"] == "gemini"]
        others = [p for p in providers_order if p["type"] != "gemini"]
        providers_order = gemini + others

    errors = []
    for provider in providers_order:
        try:
            logger.info(f"🔄 Trying {provider['name']} ({provider['model']})...")
            start = time.time()

            # Only pass images to providers that support vision
            provider_images = images if provider["type"] in ("gemini",) else None
            # Note: Some OpenAI-compat providers support vision too (Groq with llava, etc.)
            # but for free tiers, Gemini is the most reliable

            if provider["type"] == "gemini":
                result = call_gemini(provider, messages, system_prompt, provider_images)
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


def process_message(user_text: str, channel: str, thread_ts: str, user_id: str, say, files: list[dict] = None):
    """Process a message through the agent loop, with optional file attachments."""
    conv_key = get_conv_key(channel, thread_ts)

    # Process any attached files
    images = []
    file_texts = []
    if files:
        images, file_texts = process_slack_files(files)

    # Build the full user message with file content
    full_message = user_text
    if file_texts:
        file_context = "\n\n".join(file_texts)
        if full_message:
            full_message = f"{full_message}\n\n{file_context}"
        else:
            full_message = f"The user shared these files. Analyze them:\n\n{file_context}"

    # If we got images but user didn't say anything, prompt analysis
    if images and not user_text.strip():
        full_message = "The user shared an image. Describe what you see and offer to help with anything related to it."
    elif images and user_text.strip():
        full_message = f"{user_text}\n\n(The user also attached an image — analyze it in context of their message.)"

    # Store in persistent memory (text only, not image data)
    memory.add_message(conv_key, "user", full_message[:2000])

    # Get conversation history
    history = memory.get_history(conv_key, limit=MAX_HISTORY)

    # Run through agent loop
    response = agent.run_agent_loop(
        messages=history,
        call_ai_fn=lambda msgs, prompt: call_ai(msgs, prompt, images=images),
        system_prompt=SYSTEM_PROMPT,
        user_id=user_id,
    )

    # Clean up any leftover tool call syntax
    response = agent.extract_final_text(response)

    # Store response in memory
    memory.add_message(conv_key, "assistant", response[:2000])

    # Send to Slack
    say(text=truncate_for_slack(response), thread_ts=thread_ts)


@slack_app.event("message")
def handle_message(event, say):
    """Handle DMs — now with file support."""
    if event.get("bot_id") or event.get("subtype"):
        return

    channel = event["channel"]
    thread_ts = event.get("thread_ts") or event.get("ts")
    user_text = event.get("text", "").strip()
    user_id = event.get("user", "unknown")
    channel_type = event.get("channel_type", "")
    files = event.get("files", [])

    if not user_text and not files:
        return
    if channel_type not in ("im", "mpim"):
        return

    file_info = f" + {len(files)} file(s)" if files else ""
    logger.info(f"💬 DM from {user_id}: {user_text[:80]}...{file_info}")

    try:
        process_message(user_text, channel, thread_ts, user_id, say, files=files)
    except Exception as e:
        logger.error(f"❌ Error: {traceback.format_exc()}")
        say(text=f"Sorry, I hit an error: {str(e)[:200]}", thread_ts=thread_ts)


@slack_app.event("app_mention")
def handle_mention(event, say):
    """Handle @mentions in channels — now with file support."""
    channel = event["channel"]
    thread_ts = event.get("thread_ts") or event["ts"]
    user_id = event.get("user", "unknown")
    user_text = re.sub(r"<@[A-Z0-9]+>\s*", "", event.get("text", "")).strip()
    files = event.get("files", [])

    if not user_text and not files:
        say(text=f"Hey! I'm {BOT_NAME} 👋 Ask me anything — I can search the web, run code, analyze images, and more!", thread_ts=thread_ts)
        return

    file_info = f" + {len(files)} file(s)" if files else ""
    logger.info(f"📢 Mention from {user_id}: {user_text[:80]}...{file_info}")

    try:
        process_message(user_text, channel, thread_ts, user_id, say, files=files)
    except Exception as e:
        logger.error(f"❌ Error: {traceback.format_exc()}")
        say(text=f"Sorry, I hit an error: {str(e)[:200]}", thread_ts=thread_ts)


# ── Slash Commands ──

@slack_app.command("/ask")
def handle_ask(ack, command, say):
    ack()
    question = command.get("text", "").strip()
    if not question:
        say("Usage: `/ask <your question>`", channel=command["channel_id"])
        return

    response = agent.run_agent_loop(
        messages=[{"role": "user", "content": question}],
        call_ai_fn=call_ai,
        system_prompt=SYSTEM_PROMPT,
        user_id=command.get("user_id", "unknown"),
    )
    response = agent.extract_final_text(response)
    say(text=f"*Q:* {question}\n\n{truncate_for_slack(response)}", channel=command["channel_id"])


@slack_app.command("/search")
def handle_search(ack, command, say):
    ack()
    query = command.get("text", "").strip()
    if not query:
        say("Usage: `/search <what to search for>`", channel=command["channel_id"])
        return

    result = tools.run_tool("web_search", {"query": query})
    say(text=f"🔍 *Search:* {query}\n\n{truncate_for_slack(result)}", channel=command["channel_id"])


@slack_app.command("/clear")
def handle_clear(ack, command, say):
    ack()
    channel = command["channel_id"]
    conn = memory.get_db()
    try:
        conn.execute("DELETE FROM conversations WHERE conv_key LIKE ?", (f"{channel}:%",))
        conn.commit()
    finally:
        conn.close()
    say(text="🧹 Memory cleared!", channel=channel)


@slack_app.command("/providers")
def handle_providers(ack, command, say):
    ack()
    if not PROVIDERS:
        say(text="❌ No providers configured.", channel=command["channel_id"])
        return

    lines = [f"🧠 *Active AI Providers ({len(PROVIDERS)}):*\n"]
    for i, p in enumerate(PROVIDERS, 1):
        lines.append(f"{i}. *{p['name']}* — `{p['model']}`")
    lines.append(f"\n_Tried in order. If one fails, the next picks up._")
    say(text="\n".join(lines), channel=command["channel_id"])


@slack_app.command("/status")
def handle_status(ack, command, say):
    ack()
    stats = memory.get_stats()
    # Check which capabilities are available
    has_vision = any(p["type"] == "gemini" for p in PROVIDERS)
    try:
        import fitz
        has_pdf = True
    except ImportError:
        has_pdf = False

    status_text = (
        f"🤖 *{BOT_NAME} v3.0*\n\n"
        f"*Providers:* {len(PROVIDERS)} active\n"
        f"*Vision:* {'✅ Gemini' if has_vision else '❌ Add GEMINI_API_KEY'}\n"
        f"*PDF reading:* {'✅' if has_pdf else '❌ Install PyMuPDF'}\n"
        f"*Memory:*\n"
        f"  • {stats['messages']} messages stored\n"
        f"  • {stats['facts']} facts remembered\n"
        f"  • {stats['conversations']} conversations\n"
        f"  • Database: {stats['db_size_mb']} MB\n\n"
        f"*Tools:* {', '.join(tools.TOOLS.keys())}\n"
    )
    say(text=status_text, channel=command["channel_id"])


# ── Start ──
if __name__ == "__main__":
    build_providers()

    has_vision = any(p["type"] == "gemini" for p in PROVIDERS)

    if not PROVIDERS:
        logger.warning("⚠️  No AI providers configured! Add API keys to .env")

    logger.info(f"🤖 {BOT_NAME} v3.0 starting (Socket Mode)...")
    logger.info(f"   Providers: {len(PROVIDERS)}")
    logger.info(f"   Vision: {'✅ Gemini' if has_vision else '❌ no Gemini key'}")
    logger.info(f"   History: {MAX_HISTORY} messages per thread")
    logger.info(f"   Tools: {list(tools.TOOLS.keys())}")
    logger.info(f"   Memory: SQLite at {memory.DB_PATH}")
    if OPERATING_MANUAL:
        logger.info(f"   Operating manual: ✅ loaded ({len(OPERATING_MANUAL)} chars)")
    else:
        logger.info("   Operating manual: none (add operating_manual.md to enable)")

    handler = SocketModeHandler(slack_app, SLACK_APP_TOKEN)
    handler.start()
