"""
Agent tools — web search, URL fetching, Python execution, memory search.
All tools are lightweight and run within the e2-micro's 1 GB RAM.
"""

import os
import re
import json
import subprocess
import tempfile
import logging
import time

import requests as http_requests
from bs4 import BeautifulSoup

import memory
import concept_graph

logger = logging.getLogger("my-agent-mini")


# ── Tool Registry ──

TOOLS = {}


def tool(name: str, description: str, params: str):
    """Decorator to register a tool."""
    def decorator(func):
        TOOLS[name] = {
            "name": name,
            "description": description,
            "params": params,
            "func": func,
        }
        return func
    return decorator


def get_tools_description() -> str:
    """Get formatted description of all available tools for the system prompt."""
    lines = []
    for t in TOOLS.values():
        lines.append(f"  • {t['name']}({t['params']}) — {t['description']}")
    return "\n".join(lines)


# Tools that can change external state on behalf of the whole workspace
# (write to GitHub, restart the server, deploy a site publicly) are
# restricted to the bot's owner only. This matters once the bot is
# reachable by anyone in a Slack workspace/public channel, not just the
# person who set it up — without this, any Slack user could ask the bot
# to "push this to your repo" or "restart the service" and it would
# comply using the owner's own credentials.
#
# CRITICAL: run_shell and run_python are ALSO owner-only. They execute
# arbitrary code on the host VM, so any Slack user — or a prompt injection
# carried in a webpage or file the agent fetches — would otherwise get remote
# code execution on the box. There is no denylist strong enough to make shell
# access safe for untrusted users; the only correct control is not exposing it
# to them at all.
OWNER_ONLY_TOOLS = {
    # Direct code execution on the host.
    "run_shell",
    "run_python",
    "github_write_file",
    "github_create_issue",
    "restart_service",
    "deploy_static_site",
    "push_branch",
    # Scheduling is owner-only for the same reason: it commits the bot to
    # acting later, on its own, using the owner's credentials.
    "schedule_task",
    "cancel_schedule",
    # Starting a background run commits shared resources — worker threads on a
    # 1 GB box and calls against a metered paid route — to autonomous work.
    "start_background_run",
}

# Tools that need to know which conversation they were called from — the
# agent loop fills these in so the model never has to pass them.
CONTEXT_TOOLS = {
    "start_background_run": ("_user_id", "_conv_key"),
    "schedule_task": ("_user_id", "_conv_key"),
}


_OWNER_WARNED = False


def _is_owner(user_id: str) -> bool:
    owner_id = os.environ.get("OWNER_SLACK_ID", "").strip()
    if not owner_id:
        # Fail CLOSED. With no owner configured nobody may use the privileged
        # tools — including the person who installed it. This previously failed
        # OPEN, which meant shell access, GitHub writes, service restarts and
        # the /approve command were available to *any* Slack user who could
        # reach the bot. A default that is only safe when someone remembers to
        # set an environment variable is not a safe default.
        global _OWNER_WARNED
        if not _OWNER_WARNED:
            _OWNER_WARNED = True
            logger.warning(
                "OWNER_SLACK_ID is not set — owner-only tools (run_shell, "
                "run_python, GitHub writes, restart, deploy, scheduling) are "
                "DISABLED for everyone. Set it in .env to enable them."
            )
        return False
    return user_id == owner_id


def run_tool(name: str, args: dict) -> str:
    """Execute a tool by name with given arguments."""
    if name not in TOOLS:
        return f"❌ Unknown tool: {name}. Available: {', '.join(TOOLS.keys())}"

    requesting_user_id = args.pop("_requesting_user_id", None)
    if name in OWNER_ONLY_TOOLS and not _is_owner(requesting_user_id or ""):
        return (
            f"❌ Not authorized: '{name}' can only be used by the bot's owner. "
            "Ask the workspace owner to run this instead."
        )

    try:
        result = TOOLS[name]["func"](**args)
        # Truncate very long results
        if len(result) > 4000:
            result = result[:4000] + "\n\n... (truncated)"
        return result
    except Exception as e:
        return f"❌ Tool error ({name}): {str(e)[:500]}"


# ── Tool Implementations ──


@tool("web_search", "Search the web using DuckDuckGo. Returns top results.", "query")
def web_search(query: str) -> str:
    """Search the web. Tries Scrapling (browser-impersonated, most reliable on
    this VM), then the duckduckgo_search library, then plain HTML scraping."""
    result = _scrapling_search(query)
    if result:
        return result
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. **{r['title']}**\n   {r['href']}\n   {r['body'][:200]}")
        return "\n\n".join(lines)
    except ImportError:
        # Fallback: use DuckDuckGo HTML search
        return _ddg_html_search(query)
    except Exception as e:
        # Fallback on any error
        logger.warning(f"DDG search library failed: {e}, trying HTML fallback")
        return _ddg_html_search(query)


def _scrapling_search(query: str) -> str | None:
    """
    Search DuckDuckGo's HTML endpoint via Scrapling's Fetcher, which
    impersonates a real browser's TLS fingerprint — far less likely to get
    rate-limited/blocked than the plain requests fallback. Returns None if
    Scrapling isn't installed or anything fails, so callers can fall back.
    """
    try:
        from scrapling.fetchers import Fetcher
    except ImportError:
        return None
    try:
        page = Fetcher.get(
            "https://html.duckduckgo.com/html/?q=" + http_requests.utils.quote(query),
            stealthy_headers=True,
            timeout=15,
        )
        html = getattr(page, "html_content", None) or getattr(page, "body", None) or str(page)
        soup = BeautifulSoup(html, "html.parser")
        results = soup.select(".result__body")[:5]
        if not results:
            return None  # blocked or layout changed → let fallbacks try
        lines = []
        for i, r in enumerate(results, 1):
            title_el = r.select_one(".result__title")
            snippet_el = r.select_one(".result__snippet")
            link_el = r.select_one(".result__url")
            title = title_el.get_text(strip=True) if title_el else "No title"
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            link = link_el.get_text(strip=True) if link_el else ""
            lines.append(f"{i}. **{title}**\n   {link}\n   {snippet[:200]}")
        return "\n\n".join(lines)
    except Exception as e:
        logger.warning(f"Scrapling search failed: {e}, falling back")
        return None


def _ddg_html_search(query: str) -> str:
    """Fallback: scrape DuckDuckGo HTML results."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        resp = http_requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
            timeout=10,
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.select(".result__body")[:5]
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            title_el = r.select_one(".result__title")
            snippet_el = r.select_one(".result__snippet")
            link_el = r.select_one(".result__url")
            title = title_el.get_text(strip=True) if title_el else "No title"
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            link = link_el.get_text(strip=True) if link_el else ""
            lines.append(f"{i}. **{title}**\n   {link}\n   {snippet[:200]}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search failed: {str(e)[:200]}"


@tool(
    "get_weather",
    "Get current weather and today's forecast for any city, via the Open-Meteo "
    "API (free, no key). Always use this for weather questions instead of "
    "web_search — it is faster and more reliable.",
    "location",
)
def get_weather(location: str) -> str:
    """Current weather for a city using Open-Meteo geocoding + forecast."""
    try:
        geo = http_requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en"},
            timeout=10,
        ).json()
        if not geo.get("results"):
            return f"❌ Could not find a place called '{location}'."
        place = geo["results"][0]
        lat, lon = place["latitude"], place["longitude"]
        name = f"{place['name']}, {place.get('country', '')}".strip(", ")

        wx = http_requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                           "wind_speed_10m,precipitation,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto", "forecast_days": 1,
            },
            timeout=10,
        ).json()
        cur = wx["current"]
        day = wx["daily"]
        codes = {
            0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
            45: "fog", 48: "rime fog", 51: "light drizzle", 53: "drizzle",
            55: "dense drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
            71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
            81: "rain showers", 82: "violent rain showers", 95: "thunderstorm",
            96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
        }
        desc = codes.get(cur.get("weather_code"), "unknown conditions")
        return (
            f"Weather in {name} right now: {desc}, {cur['temperature_2m']}°C "
            f"(feels like {cur['apparent_temperature']}°C), humidity {cur['relative_humidity_2m']}%, "
            f"wind {cur['wind_speed_10m']} km/h, precipitation {cur['precipitation']} mm.\n"
            f"Today: {day['temperature_2m_min'][0]}–{day['temperature_2m_max'][0]}°C, "
            f"max precipitation chance {day['precipitation_probability_max'][0]}%."
        )
    except Exception as e:
        return f"❌ Weather lookup failed: {str(e)[:200]}"


# ── SSRF protection for fetch_url ──
# The agent runs on a cloud VM and will fetch any URL it is given — including
# one embedded in a webpage or file it was asked to read. Without this guard,
# fetch_url can be pointed at the cloud metadata service (169.254.169.254 on
# both GCP and AWS) to steal instance credentials, or at internal services to
# pivot inside the network. Every host is resolved to its IPs and rejected if
# any is private, loopback, link-local, reserved or multicast; redirects are
# followed manually so a redirect cannot smuggle the request onto a blocked
# host after the first check passed.

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

_SSRF_BLOCKED_HOSTNAMES = {"localhost", "metadata", "metadata.google.internal"}


def _url_host_is_safe(hostname: str) -> bool:
    if not hostname:
        return False
    host = hostname.lower()
    if host in _SSRF_BLOCKED_HOSTNAMES or host.endswith((".internal", ".local")):
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return False
    for info in infos:
        try:
            addr = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast):
            return False
    return True


def _safe_http_get(url: str, headers: dict, timeout: int = 15):
    """
    GET with SSRF protection, re-validating every redirect hop.

    Returns (response, None) or (None, error_message).
    """
    for _ in range(6):
        try:
            parsed = urlparse(url)
        except Exception:
            return None, "❌ Blocked: malformed URL."
        if parsed.scheme not in ("http", "https"):
            return None, f"❌ Blocked: unsupported scheme '{parsed.scheme}'."
        if not _url_host_is_safe(parsed.hostname or ""):
            return None, (
                "❌ Blocked: that host resolves to a private, loopback or "
                "link-local address. Internal services and cloud metadata "
                "endpoints are not reachable through this tool."
            )
        try:
            resp = http_requests.get(url, headers=headers, timeout=timeout,
                                     allow_redirects=False)
        except Exception as e:
            return None, f"Failed to fetch URL: {str(e)[:300]}"
        if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
            url = urljoin(url, resp.headers["Location"])
            continue
        return resp, None
    return None, "❌ Blocked: too many redirects."


@tool("fetch_url", "Fetch a webpage and extract its main text content.", "url")
def fetch_url(url: str) -> str:
    """Fetch a URL and extract readable text."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        resp, error = _safe_http_get(url, headers, timeout=15)
        if error:
            return error
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "json" in content_type:
            return json.dumps(resp.json(), indent=2)[:4000]

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove scripts, styles, nav, footer
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.decompose()

        # Try to find main content
        main = soup.find("main") or soup.find("article") or soup.find("body")
        if main:
            text = main.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        if not text:
            return "Could not extract text from this page."
        return text[:4000]

    except Exception as e:
        return f"Failed to fetch URL: {str(e)[:300]}"


@tool("run_python", "Execute Python code and return the output. Use for calculations, data processing, etc.", "code")
def run_python(code: str) -> str:
    """Execute Python code in a sandboxed subprocess with timeout."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            temp_path = f.name

        result = subprocess.run(
            ["python3", temp_path],
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
            cwd="/tmp",
        )

        os.unlink(temp_path)

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"

        # Same redaction as run_shell: code the model wrote can just as easily
        # print an environment variable holding a token.
        return _redact_shell_output(output).strip() or "(no output)"

    except subprocess.TimeoutExpired:
        return "❌ Code execution timed out (30s limit)"
    except Exception as e:
        return f"❌ Execution error: {str(e)[:300]}"


@tool("memory_search", "Search your past conversations for information.", "query")
def memory_search(query: str) -> str:
    """Search conversation history."""
    results = memory.search_history(query, limit=5)
    if not results:
        return "No matching conversations found."
    lines = []
    for r in results:
        time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["time"]))
        lines.append(f"[{time_str}] {r['role']}: {r['content']}")
    return "\n\n".join(lines)


@tool(
    "remember",
    'Store durable memory. category=\'decision\' for anything that must survive into future threads (priorities, roadmap, architecture/process choices, "don\'t do X yet") — decisions are never crowded out of context. category=\'fact\' (default) for casual preferences.',
    "fact, category",
)
def remember(fact: str, user_id: str = "default", category: str = "fact") -> str:
    """Store a fact or decision about the user/project."""
    category = category if category in ("fact", "decision") else "fact"
    memory.add_fact(user_id, fact, category=category)
    label = "Decision" if category == "decision" else "Noted"
    return f"✅ {label}: {fact}"


# ── Concept Graph Tools ──
# Let the agent explicitly query the entity-relationship graph for
# structured knowledge that keyword search might miss.

@tool(
    "graph_recall",
    'Search the concept graph for entities and how they connect — complements memory_search, which finds raw text. Returns entities, types, relationships, context.',
    "query",
)
def graph_recall_tool(query: str) -> str:
    """Query the concept graph for related entities."""
    results = concept_graph.recall(query, limit=8)
    if not results:
        return "No matching entities in the concept graph yet."
    lines = []
    for r in results:
        conn_str = f" ({r['relation']} → {r['connected_to']})" if r["connected_to"] else ""
        lines.append(f"• {r['entity']} [{r['type']}]{conn_str} — {r['mentions']} mentions")
        if r.get("snippet"):
            lines.append(f"  context: {r['snippet'][:150]}")
    return "\n".join(lines)


@tool(
    "graph_inspect",
    "Deep-dive on a single entity in the concept graph: see all its "
    "relationships, connection count, and when it was first/last mentioned. "
    "Use after graph_recall to explore a specific node.",
    "entity_name",
)
def graph_inspect_tool(entity_name: str) -> str:
    """Get detailed info about one entity in the concept graph."""
    ctx = concept_graph.get_entity_context(entity_name)
    if "error" in ctx:
        return ctx["error"]
    lines = [
        f"*{ctx['name']}* [{ctx['type']}]",
        f"  Mentions: {ctx['mentions']} | First seen: {ctx['first_seen']} | Last seen: {ctx['last_seen']}",
    ]
    if ctx["outgoing"]:
        lines.append("  Outgoing:")
        for e in ctx["outgoing"]:
            lines.append(f"    → {e['target']} ({e['relation']}, weight {e['weight']})")
    if ctx["incoming"]:
        lines.append("  Incoming:")
        for e in ctx["incoming"]:
            lines.append(f"    ← {e['source']} ({e['relation']}, weight {e['weight']})")
    return "\n".join(lines)


# ── Task Planner Tools ──
# For any multi-step request, create a plan first so the user can see it
# and so an interrupted task can be resumed instead of restarted from zero.

@tool("create_plan", "Create or replace the task plan for this conversation. Use for any multi-step request BEFORE starting work. steps is a list of short step descriptions.", "steps")
def create_plan_tool(steps: list, conv_key: str = "default", user_id: str = "default") -> str:
    plan = memory.create_plan(conv_key, user_id, [str(s) for s in steps])
    lines = [f"{p['step_no']}. [{p['status']}] {p['description']}" for p in plan]
    return "✅ Plan created:\n" + "\n".join(lines)


@tool("update_task", "Mark a plan step as 'in_progress', 'done', or 'blocked' after working on it.", "step_no, status")
def update_task_tool(step_no: int, status: str, conv_key: str = "default") -> str:
    ok = memory.update_task_status(conv_key, int(step_no), status)
    if not ok:
        return f"❌ No step {step_no} found in the current plan."
    return f"✅ Step {step_no} marked {status}."


@tool("list_tasks", "Show the current task plan and progress for this conversation.", "")
def list_tasks_tool(conv_key: str = "default") -> str:
    plan = memory.get_plan(conv_key)
    if not plan:
        return "No active plan for this conversation."
    lines = [f"{p['step_no']}. [{p['status']}] {p['description']}" for p in plan]
    return "\n".join(lines)


# ── Autonomy Tools — background runs and schedules ──
# These reach into runner.py / triggers.py, which import agent.py, which
# imports this module — so the imports live inside the functions to keep the
# cycle from biting at import time.

@tool(
    "start_background_run",
    "Start a long task as a background run that survives restarts and doesn't block chat. Returns a run id immediately; the result is posted when it finishes. Not for quick answers. Owner only.",
    "goal",
)
def start_background_run(goal: str, _user_id: str = "default", _conv_key: str = "") -> str:
    import runner
    channel, thread_ts = runner.conv_target(_conv_key)
    try:
        run_id = runner.enqueue_run(
            goal=goal,
            source="manual",
            owner_user_id=_user_id,
            conv_key=_conv_key,
            channel=channel,
            thread_ts=thread_ts,
            unattended=True,
        )
    except runner.RunRejected as e:
        return f"❌ {e}"
    return (
        f"✅ Queued background run #{run_id}. It runs on its own and posts the result here "
        f"when done. Tell the user the run id; check progress with run_status({run_id}). "
        "Do not wait for it — finish your reply now."
    )


@tool(
    "schedule_task",
    "Schedule work to run automatically, with no human present. when: 'every 30m', 'hourly', "
    "'daily 09:00', 'weekly mon 09:00', or a 5-field cron line. name is a short label used to "
    "cancel it later. Owner only.",
    "name, when, goal",
)
def schedule_task(name: str, when: str, goal: str,
                  _user_id: str = "default", _conv_key: str = "") -> str:
    import runner
    import triggers
    channel, thread_ts = runner.conv_target(_conv_key)
    try:
        sched = triggers.add_schedule(
            name=name, spec=when, goal=goal, owner_user_id=_user_id,
            channel=channel, thread_ts=thread_ts,
        )
    except ValueError as e:
        return f"❌ {e}"
    next_at = time.strftime("%a %Y-%m-%d %H:%M", time.localtime(sched["next_run"]))
    return (
        f"✅ Scheduled '{sched['name']}' ({sched['spec']}) — first run {next_at} (server time).\n"
        "Note: scheduled runs are unattended, so deploys, pushes and service restarts are "
        "blocked in them; they'll report what needs a human instead."
    )


@tool("list_schedules", "Show all scheduled tasks and when they next run.", "")
def list_schedules_tool() -> str:
    import triggers
    return triggers.format_schedules()


@tool("cancel_schedule", "Cancel a scheduled task by its name (or id). Owner only.", "name")
def cancel_schedule_tool(name: str) -> str:
    import triggers
    if triggers.cancel_schedule(name):
        return f"✅ Schedule '{name}' cancelled."
    return f"❌ No schedule named '{name}'. Use list_schedules to see what exists."


@tool(
    "run_status",
    "Check a background run: status, steps used, and its result if it finished. "
    "Omit run_id to list recent runs.",
    "run_id",
)
def run_status(run_id: int = 0) -> str:
    import runner
    if not run_id:
        return runner.format_runs()
    run = runner.get_run(int(run_id))
    if not run:
        return f"❌ No run #{run_id}."
    lines = [
        f"Run #{run['id']} — *{run['status']}* ({run['source']})",
        f"Goal: {run['goal'][:300]}",
        f"Steps: {run['steps_used']}/{run['max_steps']}",
    ]
    if run["error"]:
        lines.append(f"Error: {run['error'][:400]}")
    if run["result"]:
        lines.append(f"Result:\n{run['result'][:1500]}")
    return "\n".join(lines)


# Tools an unattended run may never call, even one with allow_risky set:
# each of them creates *more* autonomous work. A run that can queue runs is
# one bad reasoning step away from a fork bomb of them.
UNATTENDED_BLOCKED_TOOLS = {
    "start_background_run",
    "schedule_task",
    "cancel_schedule",
}


# ── v4 "Full Agent" Tools — shell, files, workspace ──

WORKSPACE = os.path.expanduser("~/agent_workspace")
os.makedirs(WORKSPACE, exist_ok=True)

# Commands that are never allowed (protect the VM). The agent has no
# interactive confirmation step, so high-impact commands are denied instead
# of being guessed at. Read-only checks and normal development commands remain
# available.
BLOCKED_PATTERNS = [
    r"(?<![A-Za-z0-9_])sudo(?![A-Za-z0-9_])",
    r"(?<![A-Za-z0-9_])su(?![A-Za-z0-9_])",
    r"\b(?:shutdown|reboot|halt|poweroff)\b",
    r"\bsystemctl\s+(?:stop|restart|disable|mask|poweroff|reboot)\b",
    r"\b(?:kill|pkill|killall)\b",
    r"\bmkfs(?:\.|\s)", r"\bfdisk\b", r"\bparted\b", r"\bdd\s+if=", r":\(\)\{",
    r"\brm\s+(?:-[^\s]*[rf][^\s]*\s+)?/", r"\b(?:rm|shred)\s+[^\n]*(?:\.env|memory\.db)",
    r">\s*/dev/", r"\b(?:curl|wget)\b[^\n|;]*\|\s*(?:bash|sh)\b",
]

# Never return common secret-bearing files if a command manages to read them.
SECRET_OUTPUT_PATTERNS = (
    re.compile(r"(?im)^.*(?:SLACK_BOT_TOKEN|SLACK_APP_TOKEN|API_KEY|SECRET|PASSWORD|TOKEN).*=?[^\n]*$"),
    re.compile(r"(?im)^.*(?:BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY).*\\n?(?:.*\\n?){0,3}"),
    # Credential *shapes*, not just lines that happen to mention a keyword.
    # `echo $GITHUB_TOKEN` prints a bare token with no keyword anywhere near it
    # and the patterns above sail straight past that — the most likely way a
    # real secret from this box ever reaches a Slack channel.
    re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"),      # GitHub classic
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),                  # GitHub fine-grained
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),                  # Slack
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),                         # OpenAI-style
    re.compile(r"AIza[A-Za-z0-9_-]{30,}"),                        # Google API
    re.compile(r"\bmg_[A-Za-z0-9]{20,}"),                         # merge.dev gateway
)
MAX_SHELL_OUTPUT = 3500


def _redact_shell_output(output: str) -> str:
    """Redact obvious credential lines before tool output reaches the model."""
    for pattern in SECRET_OUTPUT_PATTERNS:
        output = pattern.sub("[redacted sensitive output]", output)
    return output


@tool("run_shell", "Run a safe shell command on the server (Ubuntu). Use for read-only checks, git, curl, and development tasks. High-impact commands are blocked.", "command")
def run_shell(command: str) -> str:
    """Execute a shell command with safety checks, redaction, and timeout."""
    if not isinstance(command, str) or not command.strip():
        return "❌ Blocked: command is empty."
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return "❌ Blocked for safety: high-impact or privileged shell commands are not allowed."
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=WORKSPACE,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        output = _redact_shell_output(output).strip()
        if len(output) > MAX_SHELL_OUTPUT:
            output = output[:MAX_SHELL_OUTPUT] + "\n\n... (shell output truncated)"
        return output or "(command ran, no output)"
    except subprocess.TimeoutExpired:
        return "❌ Command timed out (60s limit)"
    except Exception as e:
        return f"❌ Shell error: {str(e)[:300]}"


@tool("write_file", "Create or overwrite a file in the agent workspace. Use for saving scripts, notes, data, reports.", "filename, content")
def write_file(filename: str, content: str) -> str:
    """Write a file inside the workspace."""
    safe_name = os.path.basename(filename)
    path = os.path.join(WORKSPACE, safe_name)
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"✅ Saved {safe_name} ({len(content)} chars) in workspace."
    except Exception as e:
        return f"❌ Write error: {str(e)[:300]}"


@tool("read_file", "Read a file from the agent workspace.", "filename")
def read_file(filename: str) -> str:
    """Read a file from the workspace."""
    safe_name = os.path.basename(filename)
    path = os.path.join(WORKSPACE, safe_name)
    try:
        with open(path, "r") as f:
            content = f.read()
        return content[:4000] if content else "(empty file)"
    except FileNotFoundError:
        return f"❌ File not found: {safe_name}. Use list_files to see what exists."
    except Exception as e:
        return f"❌ Read error: {str(e)[:300]}"


@tool("list_files", "List all files in the agent workspace.", "")
def list_files() -> str:
    """List workspace files with sizes."""
    try:
        files = sorted(os.listdir(WORKSPACE))
        if not files:
            return "Workspace is empty."
        lines = []
        for name in files:
            path = os.path.join(WORKSPACE, name)
            size = os.path.getsize(path)
            lines.append(f"  {name} ({size} bytes)")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ List error: {str(e)[:300]}"


# ── GitHub Automation Tools ──
# The VM has no git credential helper configured, so plain `git push` inside
# run_shell fails ("could not read Username"). These tools use the GitHub
# REST API with a personal access token instead — the reliable way to read
# or write repo files, and to manage issues, from this agent.

GITHUB_API = "https://api.github.com"


def _github_headers() -> dict | None:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        return None
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _github_repo_slug(owner: str, repo: str) -> tuple[str, str]:
    """Fill in defaults from env if the model omits owner/repo."""
    owner = owner or os.environ.get("GITHUB_DEFAULT_OWNER", "")
    repo = repo or os.environ.get("GITHUB_DEFAULT_REPO", "")
    return owner, repo


@tool(
    "github_read_file",
    "Read a file's content from a GitHub repo (no local git clone needed).",
    "path, owner='', repo='', branch='main'",
)
def github_read_file(path: str, owner: str = "", repo: str = "", branch: str = "main") -> str:
    headers = _github_headers()
    if not headers:
        return "❌ GITHUB_TOKEN is not configured on the server."
    owner, repo = _github_repo_slug(owner, repo)
    if not owner or not repo:
        return "❌ No owner/repo given and no GITHUB_DEFAULT_OWNER/REPO configured."
    try:
        resp = http_requests.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}",
            headers=headers,
            params={"ref": branch},
            timeout=15,
        )
        if resp.status_code == 404:
            return f"❌ File not found: {owner}/{repo}/{path} on {branch}"
        resp.raise_for_status()
        data = resp.json()
        import base64
        content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return content[:4000]
    except Exception as e:
        return f"❌ GitHub read error: {str(e)[:300]}"


@tool(
    "github_write_file",
    'Propose a one-file change to a GitHub repo: creates a branch, commits there, opens a PR — never commits to main. Returns the PR URL. Owner-only.',
    "path, content, message, owner='', repo='', base_branch='main'",
)
def github_write_file(
    path: str,
    content: str,
    message: str,
    owner: str = "",
    repo: str = "",
    base_branch: str = "main",
) -> str:
    headers = _github_headers()
    if not headers:
        return "❌ GITHUB_TOKEN is not configured on the server."
    owner, repo = _github_repo_slug(owner, repo)
    if not owner or not repo:
        return "❌ No owner/repo given and no GITHUB_DEFAULT_OWNER/REPO configured."
    try:
        import base64
        repo_url = f"{GITHUB_API}/repos/{owner}/{repo}"

        # 1. Find the base branch's current commit to branch off from.
        base_ref = http_requests.get(
            f"{repo_url}/git/ref/heads/{base_branch}", headers=headers, timeout=15
        )
        base_ref.raise_for_status()
        base_sha = base_ref.json()["object"]["sha"]

        # 2. Create a new branch for this change.
        new_branch = f"agent/{int(time.time())}-{re.sub(r'[^a-zA-Z0-9]+', '-', path)[:40].strip('-')}"
        create_ref = http_requests.post(
            f"{repo_url}/git/refs",
            headers=headers,
            json={"ref": f"refs/heads/{new_branch}", "sha": base_sha},
            timeout=15,
        )
        create_ref.raise_for_status()

        # 3. Look up the file's existing sha on the new branch (needed to update, not create).
        contents_url = f"{repo_url}/contents/{path}"
        sha = None
        existing = http_requests.get(
            contents_url, headers=headers, params={"ref": new_branch}, timeout=15
        )
        if existing.status_code == 200:
            sha = existing.json().get("sha")

        # 4. Commit the file change on the new branch.
        payload = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": new_branch,
        }
        if sha:
            payload["sha"] = sha
        put_resp = http_requests.put(contents_url, headers=headers, json=payload, timeout=20)
        put_resp.raise_for_status()

        # 5. Open a PR from the new branch into the base branch.
        pr_resp = http_requests.post(
            f"{repo_url}/pulls",
            headers=headers,
            json={
                "title": message,
                "head": new_branch,
                "base": base_branch,
                "body": f"Proposed by the agent.\n\nFile: `{path}`",
            },
            timeout=15,
        )
        pr_resp.raise_for_status()
        pr = pr_resp.json()
        return f"✅ Opened PR for review (not merged yet): {pr['html_url']}"
    except Exception as e:
        return f"❌ GitHub write error: {str(e)[:300]}"


# ── Coding Workspace: clone / edit / test / push ──
# For real multi-file work (not a single-file PR), the agent clones a repo
# locally into repos/<repo>, edits/tests it with repo_write_file/
# repo_read_file/run_shell, then push_branch ships the result as a PR —
# same "never touch main directly" safety model as github_write_file, just
# for a whole branch of commits instead of one file. The GITHUB_TOKEN is
# passed as a one-off `http.extraHeader` on the git subprocess call itself
# (not baked into the remote URL or git config), so it's never persisted to
# disk and never shows up in `git remote -v` or repo_list_files output.

REPOS_DIR = os.path.join(WORKSPACE, "repos")
os.makedirs(REPOS_DIR, exist_ok=True)


def _safe_repo_path(relpath: str) -> str | None:
    """Resolve a path under REPOS_DIR, rejecting any traversal outside it."""
    relpath = (relpath or "").strip()
    candidate = os.path.realpath(os.path.join(REPOS_DIR, relpath))
    repos_real = os.path.realpath(REPOS_DIR)
    if candidate == repos_real or candidate.startswith(repos_real + os.sep):
        return candidate
    return None


def _git_extra_header_arg() -> list[str]:
    """Build a one-off git -c http.extraHeader=... arg carrying the auth
    token, without ever writing it into the repo's git config or remote."""
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        return []
    return ["-c", f"http.extraHeader=AUTHORIZATION: bearer {token}"]


@tool(
    "clone_repo",
    'Clone a GitHub repo into repos/<repo> for multi-file work. Safe to call again — refreshes the existing clone instead of re-cloning. Private repos via GITHUB_TOKEN.',
    "repo, owner='', branch='main'",
)
def clone_repo(repo: str, owner: str = "", branch: str = "main") -> str:
    owner, repo = _github_repo_slug(owner, repo)
    if not owner or not repo:
        return "❌ No owner/repo given and no GITHUB_DEFAULT_OWNER/REPO configured."
    dest = os.path.join(REPOS_DIR, repo)
    url = f"https://github.com/{owner}/{repo}.git"
    extra = _git_extra_header_arg()
    try:
        if os.path.isdir(os.path.join(dest, ".git")):
            fetch = subprocess.run(
                ["git", *extra, "-C", dest, "fetch", "origin", branch, "--quiet"],
                capture_output=True, text=True, timeout=90,
            )
            if fetch.returncode != 0:
                return f"❌ Fetch failed: {_redact_shell_output(fetch.stderr)[:300]}"
            reset = subprocess.run(
                ["git", "-C", dest, "reset", "--hard", f"origin/{branch}", "--quiet"],
                capture_output=True, text=True, timeout=30,
            )
            if reset.returncode != 0:
                return f"❌ Reset failed: {_redact_shell_output(reset.stderr)[:300]}"
            return f"✅ Refreshed existing clone at repos/{repo} to latest {branch}."
        clone = subprocess.run(
            ["git", *extra, "clone", "--quiet", "--branch", branch, url, dest],
            capture_output=True, text=True, timeout=120,
        )
        if clone.returncode != 0:
            return f"❌ Clone failed: {_redact_shell_output(clone.stderr)[:300]}"
        subprocess.run(["git", "-C", dest, "config", "user.email", "agent@my-agent-mini"], timeout=10)
        subprocess.run(["git", "-C", dest, "config", "user.name", "My Agent"], timeout=10)
        return (
            f"✅ Cloned {owner}/{repo}@{branch} into repos/{repo}. "
            f"Use repo_read_file/repo_write_file/repo_list_files to edit, "
            f"run_shell (cd repos/{repo} && ...) to test and `git add -A && git commit -m '...'`, "
            f"then push_branch to open a PR."
        )
    except subprocess.TimeoutExpired:
        return "❌ Clone/fetch timed out."
    except Exception as e:
        return f"❌ Clone error: {str(e)[:300]}"


@tool(
    "repo_write_file",
    "Create or overwrite a file inside a cloned repo. relpath is relative to repos/, e.g. 'my-repo/src/app.py'. Run clone_repo first.",
    "relpath, content",
)
def repo_write_file(relpath: str, content: str) -> str:
    path = _safe_repo_path(relpath)
    if path is None:
        return "❌ Blocked: path escapes the repos/ workspace."
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"✅ Wrote repos/{relpath.strip('/')} ({len(content)} chars)."
    except Exception as e:
        return f"❌ Write error: {str(e)[:300]}"


@tool(
    "repo_read_file",
    "Read a file inside a cloned repo. relpath is relative to repos/, e.g. "
    "'my-repo/src/app.py'. Long files are paged: pass start_line to continue "
    "reading where the previous chunk ended (output says the next start_line).",
    "relpath, start_line=1",
)
def repo_read_file(relpath: str, start_line: int = 1) -> str:
    path = _safe_repo_path(relpath)
    if path is None:
        return "❌ Blocked: path escapes the repos/ workspace."
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        if not lines:
            return "(empty file)"
        start = max(1, int(start_line))
        out, size = [], 0
        i = start - 1
        while i < len(lines) and size < 4000:
            out.append(f"{i + 1}| {lines[i].rstrip()}")
            size += len(lines[i])
            i += 1
        chunk = "\n".join(out)
        if i < len(lines):
            chunk += f"\n... ({len(lines)} lines total — continue with start_line={i + 1})"
        return chunk
    except FileNotFoundError:
        return f"❌ File not found: repos/{relpath.strip('/')}. Use repo_list_files or clone_repo first."
    except IsADirectoryError:
        return f"❌ That's a directory, not a file: repos/{relpath.strip('/')}. Use repo_list_files."
    except Exception as e:
        return f"❌ Read error: {str(e)[:300]}"


@tool(
    "repo_edit_file",
    'Edit a file in a cloned repo by replacing an exact snippet; never rewrites the whole file. old_text must appear EXACTLY once — include enough surrounding lines to make it unique. Read the file first with repo_read_file.',
    "relpath, old_text, new_text",
)
def repo_edit_file(relpath: str, old_text: str, new_text: str) -> str:
    path = _safe_repo_path(relpath)
    if path is None:
        return "❌ Blocked: path escapes the repos/ workspace."
    try:
        with open(path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        return f"❌ File not found: repos/{relpath.strip('/')}. Use repo_write_file to create it."
    count = content.count(old_text)
    if count == 0:
        return ("❌ old_text not found in the file. Re-read the file with repo_read_file — "
                "the exact text (including whitespace) must match.")
    if count > 1:
        return (f"❌ old_text appears {count} times — include more surrounding lines "
                "so it matches exactly once.")
    with open(path, "w") as f:
        f.write(content.replace(old_text, new_text, 1))
    return f"✅ Edited repos/{relpath.strip('/')} (replaced {len(old_text)} chars with {len(new_text)})."


def _run_quality_gate(dest: str) -> tuple[bool, str]:
    """
    Quality gate for a cloned repo: compile every changed .py file, run ruff
    if installed, run pytest if a tests/ dir exists. Returns (ok, report).
    ok is False only for hard failures (syntax errors, failing tests) —
    lint warnings are reported but don't block.
    """
    report = []
    ok = True

    # Which .py files changed vs HEAD (staged, unstaged, and untracked)?
    changed = subprocess.run(
        ["git", "-C", dest, "status", "--porcelain"], capture_output=True, text=True, timeout=15
    ).stdout
    committed = subprocess.run(
        ["git", "-C", dest, "diff", "--name-only", "@{upstream}...HEAD"],
        capture_output=True, text=True, timeout=15,
    ).stdout if subprocess.run(
        ["git", "-C", dest, "rev-parse", "--abbrev-ref", "@{upstream}"],
        capture_output=True, text=True, timeout=15,
    ).returncode == 0 else ""
    py_files = set()
    for line in changed.splitlines():
        name = line[3:].strip()
        if name.endswith(".py"):
            py_files.add(name)
    for name in committed.splitlines():
        if name.strip().endswith(".py"):
            py_files.add(name.strip())

    # 1. Syntax check (hard gate).
    for name in sorted(py_files):
        full = os.path.join(dest, name)
        if not os.path.exists(full):
            continue
        r = subprocess.run(
            ["python3", "-m", "py_compile", full], capture_output=True, text=True, timeout=30
        )
        if r.returncode != 0:
            ok = False
            report.append(f"❌ SYNTAX ERROR in {name}:\n{r.stderr.strip()[:400]}")
        else:
            report.append(f"✅ {name} compiles")

    # 2. Ruff lint (soft gate — reported, not blocking).
    if py_files:
        ruff = subprocess.run(
            ["ruff", "check", *sorted(py_files)], capture_output=True, text=True,
            timeout=60, cwd=dest,
        ) if _which("ruff") else None
        if ruff is None:
            report.append("ℹ️ ruff not installed — lint skipped (pip install ruff)")
        elif ruff.returncode == 0:
            report.append("✅ ruff: clean")
        else:
            report.append(f"⚠️ ruff findings (not blocking):\n{ruff.stdout.strip()[:600]}")

    # 3. Tests (hard gate when they exist).
    if os.path.isdir(os.path.join(dest, "tests")) and _which("pytest"):
        t = subprocess.run(
            ["pytest", "-x", "-q", "tests"], capture_output=True, text=True,
            timeout=300, cwd=dest,
        )
        tail = (t.stdout + t.stderr).strip()[-600:]
        if t.returncode != 0:
            ok = False
            report.append(f"❌ TESTS FAILED:\n{tail}")
        else:
            report.append(f"✅ tests passed:\n{tail.splitlines()[-1] if tail else ''}")

    if not py_files:
        report.append("ℹ️ no changed .py files detected")
    return ok, "\n".join(report)


def _which(cmd: str) -> bool:
    import shutil
    return shutil.which(cmd) is not None


@tool(
    "repo_check",
    'Run the quality gate on a cloned repo: syntax-check changed .py files, ruff lint, pytest if tests/ exists. Run after editing, before committing.',
    "repo",
)
def repo_check(repo: str) -> str:
    dest = os.path.join(REPOS_DIR, repo.strip("/").split("/")[-1])
    if not os.path.isdir(os.path.join(dest, ".git")):
        return f"❌ No clone found at repos/{repo}. Run clone_repo first."
    try:
        ok, report = _run_quality_gate(dest)
        head = "✅ QUALITY GATE PASSED" if ok else "❌ QUALITY GATE FAILED — fix before pushing"
        return f"{head}\n{report}"
    except Exception as e:
        return f"❌ Quality gate error: {str(e)[:300]}"


@tool(
    "repo_list_files",
    "List files inside a cloned repo (or a subdirectory of it). relpath is relative to repos/, e.g. 'my-repo' or 'my-repo/src'. Skips .git internals.",
    "relpath=''",
)
def repo_list_files(relpath: str = "") -> str:
    path = _safe_repo_path(relpath) if relpath else REPOS_DIR
    if path is None:
        return "❌ Blocked: path escapes the repos/ workspace."
    if not os.path.isdir(path):
        return f"❌ Not a directory: repos/{relpath.strip('/')}. Use clone_repo first."
    lines = []
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d != ".git")
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, REPOS_DIR)
            lines.append(f"  {rel} ({os.path.getsize(full)} bytes)")
            if len(lines) >= 200:
                lines.append("  ... (truncated — narrow to a subdirectory)")
                return "\n".join(lines)
    return "\n".join(lines) if lines else "(empty)"


@tool(
    "push_branch",
    'Push committed changes from repos/<repo> as a new branch and open a PR — never touches the base branch. Commit first via run_shell. Owner-only.',
    "repo, branch_name, pr_title, owner='', pr_body='', base_branch='main'",
)
def push_branch(
    repo: str,
    branch_name: str,
    pr_title: str,
    owner: str = "",
    pr_body: str = "",
    base_branch: str = "main",
) -> str:
    headers = _github_headers()
    if not headers:
        return "❌ GITHUB_TOKEN is not configured on the server."
    owner, repo = _github_repo_slug(owner, repo)
    if not owner or not repo:
        return "❌ No owner/repo given and no GITHUB_DEFAULT_OWNER/REPO configured."
    dest = os.path.join(REPOS_DIR, repo)
    if not os.path.isdir(os.path.join(dest, ".git")):
        return f"❌ No clone found at repos/{repo}. Run clone_repo first."

    branch_name = re.sub(r"[^a-zA-Z0-9/_-]+", "-", branch_name).strip("-") or f"agent-{int(time.time())}"
    if not branch_name.startswith("agent/"):
        branch_name = f"agent/{branch_name}"

    try:
        status = subprocess.run(
            ["git", "-C", dest, "status", "--porcelain"], capture_output=True, text=True, timeout=15,
        )
        if status.stdout.strip():
            return (
                "❌ You have uncommitted changes in the clone. Commit them first: "
                f"run_shell(\"cd repos/{repo} && git add -A && git commit -m '...'\")."
            )

        # Quality gate: never push code that doesn't compile or fails tests.
        gate_ok, gate_report = _run_quality_gate(dest)
        if not gate_ok:
            return f"❌ Push blocked by quality gate — fix these first:\n{gate_report}"

        checkout = subprocess.run(
            ["git", "-C", dest, "checkout", "-B", branch_name], capture_output=True, text=True, timeout=15,
        )
        if checkout.returncode != 0:
            return f"❌ Could not create branch: {_redact_shell_output(checkout.stderr)[:300]}"

        extra = _git_extra_header_arg()
        push = subprocess.run(
            ["git", *extra, "-C", dest, "push", "--force-with-lease", "origin", f"{branch_name}:{branch_name}"],
            capture_output=True, text=True, timeout=90,
        )
        if push.returncode != 0:
            return f"❌ Push failed: {_redact_shell_output(push.stderr)[:300]}"

        repo_url = f"{GITHUB_API}/repos/{owner}/{repo}"
        pr_resp = http_requests.post(
            f"{repo_url}/pulls",
            headers=headers,
            json={
                "title": pr_title,
                "head": branch_name,
                "base": base_branch,
                "body": (pr_body or "Proposed by the agent.")
                + f"\n\n---\n**Quality gate:**\n```\n{gate_report[:1500]}\n```",
            },
            timeout=15,
        )
        if pr_resp.status_code == 422 and "already exists" in pr_resp.text:
            return f"✅ Pushed to existing branch `{branch_name}` (a PR is already open for it)."
        pr_resp.raise_for_status()
        pr = pr_resp.json()
        return f"✅ Opened PR for review (not merged yet): {pr['html_url']}"
    except subprocess.TimeoutExpired:
        return "❌ Push timed out."
    except Exception as e:
        return f"❌ Push error: {str(e)[:300]}"


@tool(
    "github_list_issues",
    "List open (or closed) issues in a GitHub repo.",
    "owner='', repo='', state='open'",
)
def github_list_issues(owner: str = "", repo: str = "", state: str = "open") -> str:
    headers = _github_headers()
    if not headers:
        return "❌ GITHUB_TOKEN is not configured on the server."
    owner, repo = _github_repo_slug(owner, repo)
    if not owner or not repo:
        return "❌ No owner/repo given and no GITHUB_DEFAULT_OWNER/REPO configured."
    try:
        resp = http_requests.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/issues",
            headers=headers,
            params={"state": state, "per_page": 20},
            timeout=15,
        )
        resp.raise_for_status()
        issues = [i for i in resp.json() if "pull_request" not in i]
        if not issues:
            return f"No {state} issues."
        lines = [f"#{i['number']} {i['title']} ({i['state']})" for i in issues]
        return "\n".join(lines)
    except Exception as e:
        return f"❌ GitHub list error: {str(e)[:300]}"


@tool(
    "github_create_issue",
    "Create a new issue in a GitHub repo.",
    "title, body='', owner='', repo=''",
)
def github_create_issue(title: str, body: str = "", owner: str = "", repo: str = "") -> str:
    headers = _github_headers()
    if not headers:
        return "❌ GITHUB_TOKEN is not configured on the server."
    owner, repo = _github_repo_slug(owner, repo)
    if not owner or not repo:
        return "❌ No owner/repo given and no GITHUB_DEFAULT_OWNER/REPO configured."
    try:
        resp = http_requests.post(
            f"{GITHUB_API}/repos/{owner}/{repo}/issues",
            headers=headers,
            json={"title": title, "body": body},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return f"✅ Created issue #{data['number']}: {data['html_url']}"
    except Exception as e:
        return f"❌ GitHub create issue error: {str(e)[:300]}"


# ── Server Administration Tools ──
# run_shell already covers most read-only checks. The one gap is safely
# restarting a service — a blanket `systemctl restart` is blocked in
# BLOCKED_PATTERNS to prevent the model from taking down anything it wants.
# restart_service instead only allows a small, explicit whitelist, so the
# agent can heal its own service (or another approved one) without being
# able to restart arbitrary units.

_ALLOWED_SERVICES = {
    s.strip() for s in os.environ.get(
        "ALLOWED_SERVICES", "my-agent.service"
    ).split(",") if s.strip()
}


@tool(
    "server_health",
    "Get a quick health snapshot of the server: uptime, disk, memory, and status of allow-listed services.",
    "",
)
def server_health() -> str:
    parts = []
    for label, cmd in [
        ("uptime", ["uptime"]),
        ("disk", ["df", "-h", "/"]),
        ("memory", ["free", "-h"]),
    ]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            parts.append(f"--- {label} ---\n{(r.stdout or r.stderr).strip()}")
        except Exception as e:
            parts.append(f"--- {label} ---\n(unavailable: {str(e)[:120]})")
    for svc in sorted(_ALLOWED_SERVICES):
        try:
            r = subprocess.run(
                ["systemctl", "is-active", svc], capture_output=True, text=True, timeout=10
            )
            parts.append(f"--- {svc} ---\n{(r.stdout or r.stderr).strip()}")
        except Exception as e:
            parts.append(f"--- {svc} ---\n(unavailable: {str(e)[:120]})")
    return "\n\n".join(parts)[:MAX_SHELL_OUTPUT]


@tool(
    "restart_service",
    "Restart one allow-listed systemd service (e.g. the agent's own service after a git pull). Any service not on the server's ALLOWED_SERVICES list is refused.",
    "service",
)
def restart_service(service: str) -> str:
    service = (service or "").strip()
    if service not in _ALLOWED_SERVICES:
        return (
            f"❌ Refused: '{service}' is not on the allow-list "
            f"({', '.join(sorted(_ALLOWED_SERVICES)) or '(empty)'}). "
            "Ask a human to restart it manually or add it to ALLOWED_SERVICES."
        )
    try:
        r = subprocess.run(
            ["sudo", "-n", "systemctl", "restart", service],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return f"❌ Restart failed ({r.returncode}): {(r.stderr or r.stdout).strip()[:300]}"
        return f"✅ Restarted {service}."
    except Exception as e:
        return f"❌ restart_service error: {str(e)[:300]}"


# ── Website Building Tools ──
# scaffold_site writes a small static site (HTML/CSS/JS) as a set of files
# in the workspace; deploy_static_site then ships that folder to Vercel
# via its REST API directly (no CLI/Node install needed on the e2-micro VM).

@tool(
    "scaffold_site",
    "Create a static website's files (e.g. index.html, style.css, script.js) as a set in a workspace subfolder, ready to preview or deploy. files is a dict of {relative_path: content}.",
    "site_name, files",
)
def scaffold_site(site_name: str, files: dict) -> str:
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", site_name).strip("-") or "site"
    site_dir = os.path.join(WORKSPACE, "sites", safe_name)
    try:
        written = []
        for rel_path, content in files.items():
            rel_path = rel_path.lstrip("/")
            full_path = os.path.normpath(os.path.join(site_dir, rel_path))
            if not full_path.startswith(os.path.normpath(site_dir)):
                continue  # refuse path traversal out of the site folder
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)
            written.append(rel_path)
        return f"✅ Scaffolded '{safe_name}' with {len(written)} files: {', '.join(written)}\nLocation: sites/{safe_name} (in the agent workspace)"
    except Exception as e:
        return f"❌ scaffold_site error: {str(e)[:300]}"


@tool(
    "deploy_static_site",
    "Deploy a scaffolded static site folder (from scaffold_site) to Vercel and return the live URL.",
    "site_name",
)
def deploy_static_site(site_name: str) -> str:
    token = os.environ.get("VERCEL_TOKEN", "").strip()
    if not token:
        return "❌ VERCEL_TOKEN is not configured on the server."
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", site_name).strip("-") or "site"
    site_dir = os.path.join(WORKSPACE, "sites", safe_name)
    if not os.path.isdir(site_dir):
        return f"❌ No such site folder: sites/{safe_name}. Run scaffold_site first."
    try:
        files_payload = []
        for root, _dirs, filenames in os.walk(site_dir):
            for fname in filenames:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, site_dir)
                with open(full_path, "r", errors="replace") as f:
                    data = f.read()
                files_payload.append({"file": rel_path, "data": data})
        if not files_payload:
            return f"❌ Site folder sites/{safe_name} is empty."

        resp = http_requests.post(
            "https://api.vercel.com/v13/deployments",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "name": safe_name,
                "files": files_payload,
                "target": "production",
                "projectSettings": {"framework": None},
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        url = data.get("url", "")
        return f"✅ Deployed! Live at: https://{url}" if url else f"✅ Deployed. Response: {json.dumps(data)[:300]}"
    except Exception as e:
        return f"❌ deploy_static_site error: {str(e)[:300]}"
