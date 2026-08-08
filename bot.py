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
import threading
from collections import OrderedDict
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
from slack_sdk.http_retry import default_retry_handlers
from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler

import memory
import tools
import agent
import governor
import runner
import triggers
import workflows
import concept_graph

# ── Logging ──
# Console (for `journalctl`/systemd) + a rotating file so history survives
# restarts and doesn't grow unbounded on the small VM disk.
LOG_DIR = Path.home() / "my-agent-mini" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "bot.log"

_log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_formatter)

from logging.handlers import RotatingFileHandler
_file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
_file_handler.setFormatter(_log_formatter)

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
logger = logging.getLogger("my-agent-mini")

# ── Health tracking ──
START_TIME = time.time()
ERROR_LOG = []  # list of (timestamp, message), capped below
MAX_ERROR_LOG = 25


def scrub(text: str) -> str:
    """Strip credentials from anything on its way to a human.

    Provider errors quote the request they failed on, so a key in a URL, a
    header dump or a JSON body ends up in Slack and in the log file. tools
    owns the redaction; this is the one-line entry point for the Slack side.
    """
    try:
        return tools._redact_shell_output(text or "")
    except Exception:
        return text or ""


def record_error(context: str, error: Exception):
    """Track recent errors in memory for /health, in addition to full logging."""
    ERROR_LOG.append((time.time(), scrub(f"{context}: {str(error)[:200]}")))
    del ERROR_LOG[:-MAX_ERROR_LOG]

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

# Slack client for file downloads and for everything a background run posts.
# The default retry handlers cover connection errors only, so a Slack 429 —
# ordinary on a busy workspace — raised instead of retrying, and runner._notify
# dropped a finished run's result with nothing but a log line.
slack_client = WebClient(
    token=SLACK_BOT_TOKEN,
    retry_handlers=default_retry_handlers() + [RateLimitErrorRetryHandler(max_retry_count=3)],
)

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

# ── Lightweight router state ──
# Inspired by OmniRoute, but kept inside this process so the 1 GB VM needs
# no second gateway service. Health is in-memory only and contains no secrets.
ROUTER_COOLDOWN_SECONDS = max(10, int(os.getenv("ROUTER_COOLDOWN_SECONDS", "90")))
# How long a call may pause for a route's token/minute window to roll before
# moving on. A short wait beats a 429 that parks the route entirely; a long
# one just moves the stall from the provider into the Slack reply.
ROUTER_MAX_TPM_WAIT_SECONDS = max(0, int(os.getenv("ROUTER_MAX_TPM_WAIT_SECONDS", "20")))
ROUTER_LOCK = threading.Lock()
PROVIDER_HEALTH = {}


def _provider_is_available(provider: dict) -> bool:
    """Return False while a provider is cooling down after a failed call."""
    with ROUTER_LOCK:
        return PROVIDER_HEALTH.get(provider["name"], {}).get("cooldown_until", 0) <= time.time()


def _record_provider_success(provider: dict) -> None:
    with ROUTER_LOCK:
        PROVIDER_HEALTH[provider["name"]] = {
            "cooldown_until": 0,
            "failures": 0,
            "last_error": "",
            "last_ok": time.time(),
        }


def _record_provider_failure(provider: dict, error: Exception) -> None:
    """Back off rate-limited/unhealthy routes instead of hammering them."""
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    with ROUTER_LOCK:
        old = PROVIDER_HEALTH.get(provider["name"], {})
        failures = int(old.get("failures", 0)) + 1
        if status == 429:
            # A rate-limited route says when it will accept traffic again.
            # Guessing instead costs availability in both directions: the flat
            # 90s sat out a token-per-minute window that clears in seconds,
            # while a genuine daily exhaustion needs far longer than 90s.
            cooldown = governor.retry_after_seconds(response, ROUTER_COOLDOWN_SECONDS)
            if cooldown is None:
                cooldown = ROUTER_COOLDOWN_SECONDS
        elif status is not None and status >= 500:
            cooldown = min(ROUTER_COOLDOWN_SECONDS, 30)
        else:
            cooldown = min(ROUTER_COOLDOWN_SECONDS, 20)
        PROVIDER_HEALTH[provider["name"]] = {
            "cooldown_until": time.time() + cooldown,
            "failures": failures,
            "last_error": f"HTTP {status}" if status else scrub(str(error))[:120],
            "last_ok": old.get("last_ok", 0),
        }


def _provider_status(provider: dict) -> str:
    with ROUTER_LOCK:
        state = PROVIDER_HEALTH.get(provider["name"], {})
    remaining = max(0, int(state.get("cooldown_until", 0) - time.time()))
    # Show the minute window alongside health. A route can be "healthy" and
    # still be one call from a 429, which is invisible without this.
    parts = []
    if governor.tpm_limit(provider["name"]):
        parts.append(
            f"{governor.tokens_last_minute(provider['name']):,}/"
            f"{governor.tpm_limit(provider['name']):,} tok/min"
        )
    if governor.rpm_limit(provider["name"]):
        parts.append(
            f"{governor.requests_last_minute(provider['name'])}/"
            f"{governor.rpm_limit(provider['name'])} req/min"
        )
    budget = (", " + ", ".join(parts)) if parts else ""
    if remaining:
        return f"cooldown ({remaining}s){budget}"
    if state.get("last_ok"):
        return f"healthy{budget}"
    return f"ready{budget}"


def build_providers():
    """Build list of available AI providers from environment variables."""
    global PROVIDERS
    PROVIDERS = []

    # Keyless best-effort route. It is first by default so stale/expired API
    # keys in an older .env cannot delay or block the free route. Set
    # ROUTER_PREFER_KEYLESS=false to use configured keys first. ROUTER_ORDER,
    # if set, names the order outright and supersedes this flag for any route
    # it mentions.
    keyless_provider = None
    if os.getenv("POLLINATIONS_ENABLED", "true").lower() not in ("0", "false", "no"):
        keyless_provider = {
            "name": "Pollinations (keyless)",
            "type": "openai_compat",
            "api_key": "",
            "model": os.getenv("POLLINATIONS_MODEL", "openai-fast"),
            "url": "https://text.pollinations.ai/{model}",
            "keyless": True,
        }

    if os.getenv("GEMINI_API_KEY"):
        PROVIDERS.append({
            "name": "Gemini",
            "type": "gemini",
            "api_key": os.environ["GEMINI_API_KEY"],
            # gemini-2.0-flash retired 2026-03-03. Free-tier access is the
            # Flash line; override with GEMINI_MODEL and check the current
            # name at ai.google.dev/gemini-api/docs/models if a call 404s.
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
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

    # Custom OpenAI-compatible endpoint — point the bot at ANY gateway or
    # local server: a free-claude-code proxy (http://host:8082/v1/chat/completions),
    # Ollama, LM Studio, vLLM, LiteLLM, etc. Set CUSTOM_LLM_URL to the full
    # /chat/completions URL; CUSTOM_LLM_API_KEY optional for local servers.
    if os.getenv("CUSTOM_LLM_URL"):
        PROVIDERS.append({
            "name": os.getenv("CUSTOM_LLM_NAME", "Custom endpoint"),
            "type": "openai_compat",
            "api_key": os.getenv("CUSTOM_LLM_API_KEY", ""),
            "model": os.getenv("CUSTOM_LLM_MODEL", "default"),
            "url": os.environ["CUSTOM_LLM_URL"],
        })

    # Merge Gateway (OpenAI-compatible paid gateway; one key can route to
    # multiple providers). Prefer the documented variable, while accepting
    # the older MERGE_API_KEY name as a backwards-compatible alias.
    merge_key = os.getenv("MERGE_GATEWAY_API_KEY") or os.getenv("MERGE_API_KEY")
    if merge_key:
        PROVIDERS.append({
            "name": "Merge Gateway",
            "type": "openai_compat",
            "api_key": merge_key,
            "model": os.getenv("MERGE_GATEWAY_MODEL", os.getenv("MERGE_MODEL", "openai/gpt-4o-mini")),
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

    if keyless_provider:
        if os.getenv("ROUTER_PREFER_KEYLESS", "true").lower() not in ("0", "false", "no"):
            PROVIDERS.insert(0, keyless_provider)
        else:
            PROVIDERS.append(keyless_provider)

    # Paid routes go last, whatever order they were registered in. Registration
    # order is just the order the blocks are written above — which had NVIDIA's
    # free tier sitting *behind* the paid gateway, so a Pollinations failure
    # would spend credit while a free route went untried. A stable sort keeps
    # every other preference (keyless first, then the free keys in order) intact.
    #
    # ROUTER_ORDER then decides the order among the free routes, because
    # registration order is not a preference anyone chose — it happened to put
    # Gemini ahead of Groq only because its block is written first. Paid stays
    # the outer key: it is a budget guard, not a preference to be overridden.
    PROVIDERS.sort(
        key=lambda p: (
            1 if governor.is_paid(p["name"]) else 0,
            governor.route_order_rank(p["name"]),
        )
    )

    paid = [p["name"] for p in PROVIDERS if governor.is_paid(p["name"])]
    logger.info(f"🧠 Loaded {len(PROVIDERS)} AI routes: {[p['name'] for p in PROVIDERS]}")
    if paid:
        logger.info(f"💳 Paid routes (tried last, capped daily): {paid}")
    if os.getenv("ROUTER_ORDER", "").strip():
        logger.info(f"   Route order from ROUTER_ORDER: {os.environ['ROUTER_ORDER']}")


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
    # The key travels in a header, never in the URL. requests puts the full
    # URL into every HTTPError message, and that message reaches the Slack
    # channel, bot.log and the /health error list — a single 400 from Gemini
    # used to publish the key to anyone in the workspace.
    headers = {"x-goog-api-key": provider["api_key"], "Content-Type": "application/json"}

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

    resp = http_requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def call_openai_compat(provider: dict, messages: list[dict], system_prompt: str, images: list[dict] = None) -> str:
    """Call any OpenAI-compatible API."""
    headers = {"Content-Type": "application/json"}
    if provider.get("api_key"):
        headers["Authorization"] = f"Bearer {provider['api_key']}"

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

    request_url = provider["url"].format(model=provider["model"])
    resp = http_requests.post(request_url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("provider returned no choices")
    content = choices[0].get("message", {}).get("content")
    if not content:
        raise RuntimeError("provider returned empty content")
    return content


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
    """Route through healthy providers, cooling down failures automatically."""
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    if not PROVIDERS:
        return "❌ No AI routes configured. Enable POLLINATIONS_ENABLED or add a provider key."

    # Vision requests prefer Gemini; keyless text routes are text-only.
    providers_order = list(PROVIDERS)
    if images:
        vision = [p for p in providers_order if p["type"] == "gemini"]
        others = [p for p in providers_order if p["type"] != "gemini"]
        providers_order = vision + others

    available = [p for p in providers_order if _provider_is_available(p)]

    # Paid routes are a real monthly budget, and an agent that starts its own
    # work can spend it while nobody is looking. Past the daily cap they drop
    # out of rotation — free routes still serve, so the bot degrades rather
    # than stopping.
    if governor.paid_budget_exhausted():
        affordable = [p for p in available if not governor.is_paid(p["name"])]
        if affordable and len(affordable) < len(available):
            logger.info(
                f"💳 Paid daily cap reached ({governor.paid_calls_today()} calls) — "
                "using free routes only"
            )
            available = affordable

    if not available:
        return "❌ All AI routes are temporarily cooling down after errors or rate limits. Try again shortly."

    # Tokens per minute, not per day, is what the agent loop actually hits: it
    # fires up to MAX_ITERATIONS calls back to back, each re-sending the whole
    # system prompt. Reordering so a route with headroom goes first turns a
    # 429 (which parks the route for its whole reset window) into a call that
    # simply succeeds somewhere else.
    input_chars = sum(len(m.get("content") or "") for m in messages) + len(system_prompt)
    est_tokens = governor.estimate_tokens(input_chars)
    available.sort(key=lambda p: governor.tpm_wait_seconds(p["name"], est_tokens))

    errors = []
    for provider in available:
        try:
            # With no route free of its minute window, waiting beats spending a
            # 429 — but only briefly, and only when nothing else can serve.
            wait = governor.tpm_wait_seconds(provider["name"], est_tokens)
            if wait > 0:
                if wait > ROUTER_MAX_TPM_WAIT_SECONDS:
                    errors.append(
                        f"• {provider['name']}: would exceed its token/minute limit "
                        f"for another {int(wait)}s"
                    )
                    continue
                logger.info(
                    f"⏳ {provider['name']} at {governor.tokens_last_minute(provider['name']):,}"
                    f"/{governor.tpm_limit(provider['name']):,} tokens this minute — "
                    f"waiting {wait:.1f}s"
                )
                time.sleep(wait)

            logger.info(f"🔄 Trying {provider['name']} ({provider['model']})...")
            start = time.time()
            provider_images = images if provider["type"] == "gemini" else None

            if provider["type"] == "gemini":
                result = call_gemini(provider, messages, system_prompt, provider_images)
            elif provider["type"] == "cohere":
                result = call_cohere(provider, messages, system_prompt)
            else:
                result = call_openai_compat(provider, messages, system_prompt, provider_images)

            _record_provider_success(provider)
            governor.record_ai_call(
                provider["name"],
                input_chars=input_chars,
                output_chars=len(result or ""),
            )
            governor.record_tokens(
                provider["name"], (input_chars + len(result or "")) // 4
            )
            logger.info(f"✅ {provider['name']} responded in {round(time.time() - start, 1)}s")
            return result

        except Exception as e:
            _record_provider_failure(provider, e)
            error_msg = scrub(str(e))[:200]
            logger.warning(f"⚠️ {provider['name']} failed; cooling down: {error_msg}")
            errors.append(f"• {provider['name']}: {error_msg}")

    error_list = scrub("\n".join(errors))
    return f"❌ All AI providers failed:\n{error_list}\n\nThe router will retry cooled-down providers automatically."


def command_channel(command: dict) -> str:
    """The channel a slash command came from.

    Reading `command_channel(command)` directly raised KeyError on a malformed
    payload, which Bolt turns into a logged 500 and the user experiences as the
    command doing nothing at all.
    """
    return command.get("channel_id") or command.get("channel", {}).get("id") or ""


def truncate_for_slack(text: str, limit: int = 3900) -> str:
    """Truncate text to fit in a single Slack message."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n_(truncated — reply to continue)_"


# ── Slack App ──
slack_app = App(token=SLACK_BOT_TOKEN)


def get_conv_key(channel: str, thread_ts: str | None) -> str:
    return f"{channel}:{thread_ts or 'main'}"


def process_message(user_text: str, channel: str, thread_ts: str, user_id: str, say,
                    files: list[dict] = None, message_ts: str = None):
    """Process a message through the agent loop, with optional file attachments."""
    conv_key = get_conv_key(channel, thread_ts)
    # React to the specific message, not the thread root: for a threaded reply
    # thread_ts points at the root, which would put the hourglass on the wrong
    # message entirely.
    react_ts = message_ts or thread_ts
    # Slack reaction gives immediate visual feedback without cluttering the chat.
    loading_reaction = os.getenv("LOADING_REACTION", "hourglass_flowing_sand").strip()
    try:
        if loading_reaction:
            slack_client.reactions_add(channel=channel, timestamp=react_ts, name=loading_reaction)
    except Exception as e:
        logger.debug(f"Could not add loading reaction: {e}")

    def remove_loading_reaction():
        try:
            if loading_reaction:
                slack_client.reactions_remove(channel=channel, timestamp=react_ts, name=loading_reaction)
        except Exception as e:
            logger.debug(f"Could not remove loading reaction: {e}")

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

    # Extract entities/relationships into the concept graph (fast, inline)
    try:
        concept_graph.extract_and_store(full_message, conv_key)
    except Exception as e:
        logger.debug(f"Concept graph extraction skipped: {e}")

    # Get conversation history
    history = memory.get_history(conv_key, limit=MAX_HISTORY)

    # Pull in older messages from this same thread that are topically
    # relevant to the current message but fell outside the recent window —
    # cheap keyword-overlap retrieval, no embeddings needed on a 1 GB VM.
    relevant = memory.search_relevant(conv_key, full_message, exclude_recent=MAX_HISTORY, limit=5)
    if relevant:
        context_lines = [f"- ({r['role']}) {r['content']}" for r in relevant]
        history = [{
            "role": "user",
            "content": (
                "[RELEVANT EARLIER CONTEXT from this same conversation, "
                "outside the recent window below — use only if actually helpful]\n"
                + "\n".join(context_lines)
            ),
        }] + history

    # Cross-thread memory: pull in relevant snippets and thread summaries
    # from OTHER conversations, so a brand-new thread can recall decisions
    # and discussions from weeks ago (durable memory of our whole history).
    cross = memory.search_all_relevant(full_message, exclude_conv_key=conv_key,
                                       limit=4, scope_channel=channel)
    if cross:
        cross_lines = []
        for r in cross:
            when = time.strftime("%Y-%m-%d", time.localtime(r["time"]))
            if r["kind"] == "summary":
                cross_lines.append(f"- [thread summary, {when}] {r['content']}")
            else:
                cross_lines.append(f"- [{r['role']}, {when}] {r['content']}")
        history = [{
            "role": "user",
            "content": (
                "[MEMORY FROM OTHER PAST CONVERSATIONS — possibly relevant "
                "history from separate threads. Use it to stay consistent with "
                "past decisions; ignore if not relevant]\n"
                + "\n".join(cross_lines)
            ),
        }] + history

    # Concept graph recall: surface related entities and connections that
    # keyword search might miss (e.g. "what tools does this project use?"
    # finds the project node and walks its edges to technology nodes).
    try:
        graph_context = concept_graph.recall(full_message, limit=5)
        graph_text = concept_graph.format_recall_for_prompt(graph_context)
        if graph_text:
            history = [{"role": "user", "content": graph_text}] + history
    except Exception as e:
        logger.debug(f"Concept graph recall skipped: {e}")

    # Run through agent loop — surface plan creation/updates as their own
    # Slack message so the user actually sees the numbered plan appear,
    # instead of it only living inside the model's hidden reasoning.
    def _post_plan_update(tool_name, tool_result):
        say(text=tool_result, thread_ts=thread_ts)

    response = agent.run_agent_loop(
        messages=history,
        call_ai_fn=lambda msgs, prompt: call_ai(msgs, prompt, images=images),
        system_prompt=SYSTEM_PROMPT,
        user_id=user_id,
        conv_key=conv_key,
        on_tool_call=_post_plan_update,
    )

    # Clean up any leftover tool call syntax
    response = agent.extract_final_text(response)

    # Store response in memory
    memory.add_message(conv_key, "assistant", response[:2000])

    # Extract entities from the bot's own response too
    try:
        concept_graph.extract_and_store(response, conv_key)
    except Exception:
        pass

    # Send to Slack, then remove the temporary thinking indicator.
    try:
        say(text=truncate_for_slack(response), thread_ts=thread_ts)
    finally:
        remove_loading_reaction()

    # Rolling thread summary: every SUMMARY_EVERY new messages, condense the
    # thread into a short digest (one cheap extra AI call, in the background
    # so it never delays the reply). Summaries feed cross-thread memory.
    try:
        _maybe_summarize_thread(conv_key, user_id)
    except Exception as e:
        logger.debug(f"Thread summarization skipped: {e}")


SUMMARY_EVERY = int(os.getenv("SUMMARY_EVERY", "12"))


def _maybe_summarize_thread(conv_key: str, user_id: str):
    """Kick off a background rolling summary when enough new messages piled up."""
    total = memory.count_messages(conv_key)
    old_summary, last_count = memory.get_summary_state(conv_key)
    if total - last_count < SUMMARY_EVERY:
        return

    def _worker():
        try:
            recent = memory.get_history(conv_key, limit=30)
            convo_text = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in recent)
            prompt = (
                "Summarize this conversation thread into a compact digest (max 150 words). "
                "Keep: decisions made, priorities stated, tasks completed or planned, "
                "important facts and preferences, exact wording of key instructions. "
                "Drop: greetings, filler. Write in terse note form.\n\n"
                + (f"PREVIOUS SUMMARY (merge into the new one):\n{old_summary}\n\n" if old_summary else "")
                + f"CONVERSATION:\n{convo_text}"
            )
            summary = call_ai([{"role": "user", "content": prompt}],
                              "You write terse, accurate conversation digests.")
            if summary and not summary.startswith("❌"):
                memory.save_thread_summary(conv_key, user_id, summary, total)
                # Deeper LLM-assisted graph extraction from the summary
                concept_graph.extract_from_summary_async(
                    summary, conv_key,
                    call_ai_fn=lambda msgs, prompt: call_ai(msgs, prompt),
                )
        except Exception as e:
            logger.warning(f"Thread summary failed for {conv_key}: {e}")

    threading.Thread(target=_worker, daemon=True).start()


# ── Delivered-once ──
# Slack redelivers an event after a socket reconnect, and a message that
# @-mentions the bot inside a DM arrives twice — once as message.im, once as
# app_mention. Neither listener could tell, so the whole agent loop ran again:
# a second reply, a second set of tool calls, double the AI spend. One channel
# + timestamp is the message's identity; first listener to claim it wins.
_SEEN_EVENTS: "OrderedDict[tuple, float]" = OrderedDict()
_SEEN_LOCK = threading.Lock()
SEEN_EVENT_TTL = 900          # a redelivery arrives in seconds, not minutes
SEEN_EVENT_MAX = 1024


def _claim_event(channel: str, ts: str) -> bool:
    """True if this message is ours to handle; False if it was already taken."""
    if not ts:
        return True
    key = (channel, ts)
    now = time.time()
    with _SEEN_LOCK:
        cutoff = now - SEEN_EVENT_TTL
        while _SEEN_EVENTS and next(iter(_SEEN_EVENTS.values())) < cutoff:
            _SEEN_EVENTS.popitem(last=False)
        if key in _SEEN_EVENTS:
            logger.info(f"↩️ Ignoring duplicate delivery of {channel}/{ts}")
            return False
        _SEEN_EVENTS[key] = now
        while len(_SEEN_EVENTS) > SEEN_EVENT_MAX:
            _SEEN_EVENTS.popitem(last=False)
    return True


@slack_app.event("message")
def handle_message(event, say):
    """Handle DMs — now with file support."""
    # `file_share` is an ordinary message that happens to carry an upload, and
    # the whole file pipeline below only ever runs for it. The other subtypes
    # (message_changed, message_deleted, channel_join …) really are noise.
    if event.get("bot_id") or event.get("subtype") not in (None, "", "file_share"):
        return

    channel = event["channel"]
    thread_ts = event.get("thread_ts") or event.get("ts")
    msg_ts = event.get("ts")
    user_text = event.get("text", "").strip()
    user_id = event.get("user", "unknown")
    channel_type = event.get("channel_type", "")
    files = event.get("files", [])

    if not user_text and not files:
        return
    if channel_type not in ("im", "mpim"):
        return
    if not _claim_event(channel, msg_ts):
        return

    file_info = f" + {len(files)} file(s)" if files else ""
    logger.info(f"💬 DM from {user_id}: {user_text[:80]}...{file_info}")

    try:
        process_message(user_text, channel, thread_ts, user_id, say, files=files, message_ts=msg_ts)
    except Exception as e:
        logger.error(f"❌ Error: {traceback.format_exc()}")
        record_error("process_message", e)
        try:
            slack_client.reactions_remove(channel=channel, timestamp=msg_ts, name=os.getenv("LOADING_REACTION", "hourglass_flowing_sand"))
        except Exception:
            pass
        try:
            say(text=f"Sorry, I hit an error: {scrub(str(e))[:200]}", thread_ts=thread_ts)
        except Exception as post_error:
            # The apology failed the same way the reply did (channel not
            # joined, Slack degraded). Raising here only produces a second
            # traceback in the log and still tells the user nothing.
            logger.error(f"❌ Could not deliver the error message either: {post_error}")


@slack_app.event("app_mention")
def handle_mention(event, say):
    """Handle @mentions in channels — now with file support."""
    # Same guard handle_message has. Without it, any app that posts text
    # containing this bot's handle — an alerting webhook, a CI notifier,
    # another assistant — starts a full agent loop, and two such bots can
    # keep each other going until a budget stops them.
    if event.get("bot_id"):
        return
    channel = event["channel"]
    thread_ts = event.get("thread_ts") or event["ts"]
    msg_ts = event.get("ts")
    user_id = event.get("user", "unknown")
    # `<@U123>` and Slack's labelled `<@U123|name>` are the same mention.
    user_text = re.sub(r"<@[A-Z0-9]+(?:\|[^>]*)?>\s*", "", event.get("text", "")).strip()
    files = event.get("files", [])

    if not _claim_event(channel, msg_ts):
        return

    if not user_text and not files:
        say(text=f"Hey! I'm {BOT_NAME} 👋 Ask me anything — I can search the web, run code, analyze images, and more!", thread_ts=thread_ts)
        return

    file_info = f" + {len(files)} file(s)" if files else ""
    logger.info(f"📢 Mention from {user_id}: {user_text[:80]}...{file_info}")

    try:
        process_message(user_text, channel, thread_ts, user_id, say, files=files, message_ts=msg_ts)
    except Exception as e:
        logger.error(f"❌ Error: {traceback.format_exc()}")
        record_error("process_message", e)
        try:
            slack_client.reactions_remove(channel=channel, timestamp=msg_ts, name=os.getenv("LOADING_REACTION", "hourglass_flowing_sand"))
        except Exception:
            pass
        try:
            say(text=f"Sorry, I hit an error: {scrub(str(e))[:200]}", thread_ts=thread_ts)
        except Exception as post_error:
            # The apology failed the same way the reply did (channel not
            # joined, Slack degraded). Raising here only produces a second
            # traceback in the log and still tells the user nothing.
            logger.error(f"❌ Could not deliver the error message either: {post_error}")


# ── Slash Commands ──

@slack_app.command("/ask")
def handle_ask(ack, command, say):
    ack()
    question = command.get("text", "").strip()
    if not question:
        say("Usage: `/ask <your question>`", channel=command_channel(command))
        return

    response = agent.run_agent_loop(
        messages=[{"role": "user", "content": question}],
        call_ai_fn=call_ai,
        system_prompt=SYSTEM_PROMPT,
        user_id=command.get("user_id", "unknown"),
    )
    response = agent.extract_final_text(response)
    say(text=f"*Q:* {question}\n\n{truncate_for_slack(response)}", channel=command_channel(command))


@slack_app.command("/search")
def handle_search(ack, command, say):
    ack()
    query = command.get("text", "").strip()
    if not query:
        say("Usage: `/search <what to search for>`", channel=command_channel(command))
        return

    result = tools.run_tool("web_search", {"query": query})
    say(text=f"🔍 *Search:* {query}\n\n{truncate_for_slack(result)}", channel=command_channel(command))


@slack_app.command("/clear")
def handle_clear(ack, command, say):
    ack()
    if not tools._is_owner(command.get("user_id", "")):
        say(text="❌ Only the bot's owner can clear memory.", channel=command_channel(command))
        return
    channel = command_channel(command)
    # Thread summaries are memory too: leaving them behind meant a "cleared"
    # channel could still quote the cleared conversation back, via the rolling
    # digest and cross-thread recall. Project memory (decisions) is per-user
    # and deliberately survives — so the reply now says exactly what went.
    conn = memory.get_db()
    try:
        messages = conn.execute(
            "DELETE FROM conversations WHERE conv_key LIKE ?", (f"{channel}:%",)
        ).rowcount
        summaries = conn.execute(
            "DELETE FROM thread_summaries WHERE conv_key LIKE ?", (f"{channel}:%",)
        ).rowcount
        conn.commit()
    finally:
        conn.close()
    say(
        text=(f"🧹 Cleared this channel's memory: {messages} message(s) and "
              f"{summaries} thread summar{'y' if summaries == 1 else 'ies'}.\n"
              "_Other channels are untouched, and durable project memory "
              "(decisions you asked me to remember) is kept — ask me to forget "
              "a specific decision if you want that gone too._"),
        channel=channel,
    )


@slack_app.command("/workflow")
def handle_workflow(ack, command, say):
    """`/workflow` to list the recurring jobs, `/workflow start <name>` to schedule one."""
    ack()
    channel = command_channel(command)
    args = (command.get("text") or "").strip().split()

    if not args or args[0] in ("list", "help"):
        say(text=workflows.describe(), channel=channel)
        return

    if args[0] != "start" or len(args) < 2:
        say(text="Usage: `/workflow` or `/workflow start <name>`", channel=channel)
        return

    # Scheduling commits the bot to acting later with the owner's credentials,
    # which is the same reason schedule_task is owner-only.
    if not tools._is_owner(command["user_id"]):
        say(text="❌ Only the bot's owner can schedule workflows.", channel=channel)
        return

    say(
        text=workflows.start(args[1], owner_user_id=command["user_id"], channel=channel),
        channel=channel,
    )


@slack_app.command("/providers")
def handle_providers(ack, command, say):
    ack()
    if not PROVIDERS:
        say(text="❌ No providers configured.", channel=command_channel(command))
        return

    lines = [f"🧠 *AI Router ({len(PROVIDERS)} routes):*\n"]
    for i, p in enumerate(PROVIDERS, 1):
        kind = "keyless best-effort" if p.get("keyless") else "your key"
        lines.append(f"{i}. *{p['name']}* — `{p['model']}` ({kind}; {_provider_status(p)})")
    lines.append("\n_Healthy routes are tried first. Rate-limited routes cool down automatically._")
    say(text="\n".join(lines), channel=command_channel(command))


@slack_app.command("/runs")
def handle_runs(ack, command, say):
    """Show background runs, or cancel one with `/runs cancel <id>`."""
    ack()
    channel = command_channel(command)
    text = command.get("text", "").strip().split()

    if text and text[0] == "cancel":
        # Cancelling is destructive and the run is the owner's work, started
        # with the owner's credentials. Reading the list is open; stopping one
        # is not.
        if not tools._is_owner(command.get("user_id", "")):
            say(text="❌ Only the bot's owner can cancel a run.", channel=channel)
            return
        if len(text) < 2 or not text[1].isdigit():
            say(text="Usage: `/runs cancel <run id>`", channel=channel)
            return
        run_id = int(text[1])
        if runner.cancel_run(run_id):
            say(text=f"🛑 Run #{run_id} cancelled (it stops at its next step).", channel=channel)
        else:
            say(text=f"Run #{run_id} isn't queued or running.", channel=channel)
        return

    is_owner = tools._is_owner(command.get("user_id", ""))

    if text and text[0].isdigit():
        if not is_owner:
            say(text="❌ Run detail is owner-only. `/runs` shows status without goals.",
                channel=channel)
            return
        say(text=tools.run_tool("run_status", {"run_id": int(text[0])}), channel=channel)
        return

    say(
        text="🏃 *Background runs*\n\n" + runner.format_runs(show_goals=is_owner)
        + "\n\n_`/runs <id>` for detail, `/runs cancel <id>` to stop one._",
        channel=channel,
    )


@slack_app.command("/schedules")
def handle_schedules(ack, command, say):
    """Show scheduled tasks, or cancel one with `/schedules cancel <name>`."""
    ack()
    channel = command_channel(command)
    text = command.get("text", "").strip().split(maxsplit=1)

    if text and text[0] == "cancel":
        # Same rule as schedule_task, which is owner-only to create: deleting
        # one is not recoverable from Slack.
        if not tools._is_owner(command.get("user_id", "")):
            say(text="❌ Only the bot's owner can cancel a schedule.", channel=channel)
            return
        if len(text) < 2:
            say(text="Usage: `/schedules cancel <name>`", channel=channel)
            return
        if triggers.cancel_schedule(text[1].strip()):
            say(text=f"✅ Schedule '{text[1].strip()}' cancelled.", channel=channel)
        else:
            say(text=f"❌ No schedule named '{text[1].strip()}'.", channel=channel)
        return

    say(
        text="⏰ *Scheduled tasks*\n\n"
        + triggers.format_schedules(show_goals=tools._is_owner(command.get("user_id", "")))
        + "\n\n_`/schedules cancel <name>` to remove one._",
        channel=channel,
    )


@slack_app.command("/approvals")
def handle_approvals(ack, command, say):
    """Show what the agent is waiting on a human to authorise."""
    ack()
    say(text="🖐️ *Waiting for approval*\n\n" + governor.format_approvals(),
        channel=command_channel(command))


def _decide_approval(command, say, approved: bool):
    channel = command_channel(command)
    parts = command.get("text", "").strip().split(maxsplit=1)
    verb = "approve" if approved else "deny"
    if not parts or not parts[0].isdigit():
        say(text=f"Usage: `/{verb} <approval id>`" + ("" if approved else " `<optional reason>`"),
            channel=channel)
        return

    approval_id = int(parts[0])
    note = parts[1] if len(parts) > 1 else ""
    user_id = command.get("user_id", "")

    # Only the owner decides. An approval queue anyone can answer is not a
    # control, and the runs behind it act with the owner's credentials.
    if not tools._is_owner(user_id):
        say(text="❌ Only the bot's owner can approve or deny.", channel=channel)
        return

    decided = governor.decide(approval_id, approved, decided_by=user_id, note=note)
    if not decided:
        say(text=f"❌ Approval #{approval_id} isn't pending (already decided, or no such id).",
            channel=channel)
        return

    resumed = runner.resume_after_decision(decided)
    word = "Approved" if approved else "Denied"
    say(
        text=f"{'✅' if approved else '🚫'} {word} `{decided['tool']}` for run "
             f"#{decided['run_id']}."
             + (" The run is back in the queue." if resumed
                else " (The run was no longer waiting.)"),
        channel=channel,
    )


@slack_app.command("/approve")
def handle_approve(ack, command, say):
    ack()
    _decide_approval(command, say, True)


@slack_app.command("/deny")
def handle_deny(ack, command, say):
    ack()
    _decide_approval(command, say, False)


@slack_app.command("/costs")
def handle_costs(ack, command, say):
    """AI call accounting, with paid routes tracked separately."""
    ack()
    say(text=governor.format_usage(), channel=command_channel(command))


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

    graph_stats = concept_graph.get_graph_stats()
    status_text = (
        f"🤖 *{BOT_NAME} v3.0*\n\n"
        f"*AI routes:* {len(PROVIDERS)} active (automatic fallback + cooldowns)\n"
        f"*Vision:* {'✅ Gemini' if has_vision else '❌ Add GEMINI_API_KEY'}\n"
        f"*PDF reading:* {'✅' if has_pdf else '❌ Install PyMuPDF'}\n"
        f"*Memory:*\n"
        f"  • {stats['messages']} messages stored\n"
        f"  • {stats['facts']} facts remembered\n"
        f"  • {stats['conversations']} conversations\n"
        f"  • Database: {stats['db_size_mb']} MB\n"
        f"*Concept graph:* {graph_stats['entities']} entities, "
        f"{graph_stats['edges']} connections\n\n"
        f"*Tools:* {', '.join(tools.TOOLS.keys())}\n"
    )
    say(text=status_text, channel=command_channel(command))


@slack_app.command("/health")
def handle_health(ack, command, say):
    """Deep operational health check — uptime, provider health, errors, disk."""
    ack()

    uptime_s = int(time.time() - START_TIME)
    days, rem = divmod(uptime_s, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    uptime_str = f"{days}d {hours}h {minutes}m" if days else f"{hours}h {minutes}m"

    healthy = [p["name"] for p in PROVIDERS if _provider_status(p) == "healthy" or _provider_status(p) == "ready"]
    cooling = [p["name"] for p in PROVIDERS if _provider_status(p).startswith("cooldown")]

    stats = memory.get_stats()

    # Disk space where the DB/logs live
    try:
        import shutil
        disk = shutil.disk_usage(Path.home() / "my-agent-mini")
        disk_free_mb = round(disk.free / 1024 / 1024)
        disk_total_mb = round(disk.total / 1024 / 1024)
        disk_str = f"{disk_free_mb} MB free / {disk_total_mb} MB total"
    except Exception:
        disk_str = "unavailable"

    log_size_kb = round(LOG_FILE.stat().st_size / 1024, 1) if LOG_FILE.exists() else 0

    # Errors quote whatever failed — request URLs, file paths, provider
    # payloads. Scrubbed, but still operator detail, so only the owner sees it.
    recent_errors = ERROR_LOG[-5:] if tools._is_owner(command.get("user_id", "")) else []
    if recent_errors:
        err_lines = "\n".join(
            f"  • {time.strftime('%m-%d %H:%M', time.localtime(ts))} — {msg}"
            for ts, msg in recent_errors
        )
    elif not tools._is_owner(command.get("user_id", "")):
        err_lines = f"  _({len(ERROR_LOG)} tracked — owner only)_"
    else:
        err_lines = "  none 🎉"

    health_text = (
        f"🩺 *{BOT_NAME} Health*\n\n"
        f"*Uptime:* {uptime_str}\n"
        f"*AI routes:* {len(healthy)}/{len(PROVIDERS)} healthy"
        + (f", cooling down: {', '.join(cooling)}" if cooling else "") + "\n"
        f"*Memory DB:* {stats['messages']} msgs, {stats['facts']} facts, "
        f"{stats['open_tasks']} open plan step(s), {stats['db_size_mb']} MB\n"
        f"*Concept graph:* {concept_graph.get_graph_stats()['entities']} entities, "
        f"{concept_graph.get_graph_stats()['edges']} connections\n"
        f"*Autonomy:* {len(runner.list_runs(limit=99, status='running'))} run(s) executing, "
        f"{len(runner.list_runs(limit=99, status='queued'))} queued, "
        f"{len(runner.list_runs(limit=99, status=runner.AWAITING_APPROVAL))} awaiting approval, "
        f"{len(triggers.list_schedules(include_disabled=False))} active schedule(s)\n"
        f"*Paid AI calls today:* {governor.paid_calls_today()}"
        + (f"/{governor.paid_daily_limit()}" if governor.paid_daily_limit() else " (uncapped)")
        + "\n"
        f"*Disk:* {disk_str}\n"
        f"*Log file:* {log_size_kb} KB (rotates at 5 MB, keeps 3 backups)\n"
        f"*Errors in this process ({len(ERROR_LOG)} tracked, last 5):*\n{err_lines}\n"
    )
    say(text=health_text, channel=command_channel(command))


# ── Autonomy wiring ──
# The run engine and scheduler are deliberately Slack-agnostic: they get an
# AI backend and a "post this somewhere" callback injected here, which is
# what keeps them testable without a Slack workspace.

def post_run_message(channel: str, thread_ts: str, text: str):
    """Where a background run reports back to."""
    slack_client.chat_postMessage(
        channel=channel,
        thread_ts=thread_ts or None,
        text=truncate_for_slack(text),
    )


def start_autonomy() -> tuple[int, bool]:
    """Bring up background workers + the scheduler. Returns (workers, scheduler_on)."""
    runner.configure(
        call_ai_fn=lambda msgs, prompt: call_ai(msgs, prompt),
        system_prompt=SYSTEM_PROMPT,
        post_message=post_run_message,
    )
    workers = runner.start_workers()
    scheduler_on = triggers.start_scheduler()

    # Context budgeting is invisible until a model rejects the request, so
    # state it at boot. The system prompt is sent on every call and dwarfs the
    # transcript budget; a route with too small a window fails on step one.
    prompt_chars = len(agent.get_agent_system_prompt(SYSTEM_PROMPT))
    worst_case = prompt_chars + runner.context_limit_chars()
    logger.info(
        f"   Context: system prompt {prompt_chars:,} chars (~{prompt_chars // 4:,} tok) "
        f"+ transcript budget {runner.context_limit_chars():,} chars "
        f"→ worst case ~{worst_case // 4:,} tokens/call; routes need "
        f"~{((worst_case // 4) // 1000) + 2}k context"
    )
    return workers, scheduler_on


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

    if not os.getenv("OWNER_SLACK_ID", "").strip():
        logger.warning(
            "⚠️  OWNER_SLACK_ID is not set — owner-only tools (run_shell, run_python, "
            "GitHub writes, restart, deploy, scheduling, background runs) are DISABLED "
            "for everyone, and /approve will refuse. Set it in .env to your Slack "
            "member ID (profile → More → Copy member ID)."
        )

    workers, scheduler_on = start_autonomy()
    logger.info(f"   Run workers: {workers} (max {runner.max_steps_default()} steps, "
                f"{runner.max_seconds_default()}s per run)")
    logger.info(f"   Scheduler: {'✅ on' if scheduler_on else '❌ off'}, "
                f"{len(triggers.list_schedules(include_disabled=False))} active schedule(s)")

    handler = SocketModeHandler(slack_app, SLACK_APP_TOKEN)
    handler.start()
