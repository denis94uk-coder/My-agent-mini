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
