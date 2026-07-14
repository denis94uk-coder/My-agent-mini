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


# Tools that can change external state on behalf of the whole workspace
# (write to GitHub, restart the server, deploy a site publicly) are
# restricted to the bot's owner only. This matters once the bot is
# reachable by anyone in a Slack workspace/public channel, not just the
# person who set it up — without this, any Slack user could ask the bot
# to "push this to your repo" or "restart the service" and it would
# comply using the owner's own credentials.
OWNER_ONLY_TOOLS = {
    "github_write_file",
    "github_create_issue",
    "restart_service",
    "deploy_static_site",
    "push_branch",
}


def _is_owner(user_id: str) -> bool:
    owner_id = os.environ.get("OWNER_SLACK_ID", "").strip()
    if not owner_id:
        # No owner configured: fail open only for single-user/dev setups.
        # Set OWNER_SLACK_ID before exposing the bot beyond yourself.
        return True
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


@tool(
    "remember",
    "Store durable memory for future conversations. Use category='decision' for "
    "things that must survive into future threads — stated priorities, roadmap "
    "items, architecture/process choices, explicit instructions ('don't do X yet'). "
    "Use category='fact' (default) for casual preferences/details. Decisions are "
    "never crowded out of context the way plain facts are.",
    "fact, category",
)
def remember(fact: str, user_id: str = "default", category: str = "fact") -> str:
    """Store a fact or decision about the user/project."""
    category = category if category in ("fact", "decision") else "fact"
    memory.add_fact(user_id, fact, category=category)
    label = "Decision" if category == "decision" else "Noted"
    return f"✅ {label}: {fact}"


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
    "Propose a change to a file in a GitHub repo. Creates a new branch, commits the change there, and opens a pull request against the base branch for human review — it never commits straight to main. Returns the PR URL. Owner-only tool.",
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
    "Clone a GitHub repo into the coding workspace (repos/<repo>) so you can read, edit, and test multiple files with repo_read_file/repo_write_file/run_shell. Safe to call again later — refreshes the existing clone to the latest branch instead of re-cloning. Works for private repos via GITHUB_TOKEN.",
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
    "Read a file inside a cloned repo. relpath is relative to repos/, e.g. 'my-repo/src/app.py'.",
    "relpath",
)
def repo_read_file(relpath: str) -> str:
    path = _safe_repo_path(relpath)
    if path is None:
        return "❌ Blocked: path escapes the repos/ workspace."
    try:
        with open(path, "r") as f:
            content = f.read()
        return content[:4000] if content else "(empty file)"
    except FileNotFoundError:
        return f"❌ File not found: repos/{relpath.strip('/')}. Use repo_list_files or clone_repo first."
    except IsADirectoryError:
        return f"❌ That's a directory, not a file: repos/{relpath.strip('/')}. Use repo_list_files."
    except Exception as e:
        return f"❌ Read error: {str(e)[:300]}"


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
    "Push locally committed changes from a cloned repo (repos/<repo>) as a new branch and open a pull request for human review — never touches the base branch directly. Commit your changes first via run_shell (cd repos/<repo> && git add -A && git commit -m '...'). Owner-only tool.",
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
                "body": pr_body or "Proposed by the agent.",
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
