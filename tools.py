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


def run_tool(name: str, args: dict) -> str:
    """Execute a tool by name with given arguments."""
    if name not in TOOLS:
        return f"❌ Unknown tool: {name}. Available: {', '.join(TOOLS.keys())}"
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
    """Search DuckDuckGo and return results."""
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


@tool("fetch_url", "Fetch a webpage and extract its main text content.", "url")
def fetch_url(url: str) -> str:
    """Fetch a URL and extract readable text."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        resp = http_requests.get(url, headers=headers, timeout=15, allow_redirects=True)
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

        return output.strip() or "(no output)"

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


@tool("remember", "Store a fact or note about the user for future reference.", "fact")
def remember(fact: str, user_id: str = "default") -> str:
    """Store a fact about the user."""
    memory.add_fact(user_id, fact)
    return f"✅ Noted: {fact}"


# ── Task Planner Tools ──
# For any multi-step request, create a plan first so the user can see it
# and so an interrupted task can be resumed instead of restarted from zero.

@tool("create_plan", "Create or replace the task plan for this conversation. Use for any multi-step request BEFORE starting work. steps is a list of short step descriptions.", "steps")
def create_plan_tool(steps: list, conv_key: str = "default") -> str:
    plan = memory.create_plan(conv_key, "default", [str(s) for s in steps])
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
