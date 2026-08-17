"""
Model Context Protocol client — the streamable-HTTP transport only.

Why this is hand-rolled rather than `pip install mcp`: the official SDK pulls
26 packages and 9.1 MB of wheels (pydantic-core, starlette, uvicorn,
cryptography, opentelemetry) and is async throughout, into a process whose
entire HTTP surface is `requests` and whose deployment target is a 1 GB box.
Most of that weight is the *server* half and the stdio transport, neither of
which this agent needs. What we actually need is four JSON-RPC methods over
HTTP POST, which is the file below and no new dependencies.

The cost of that choice is honest: we own protocol correctness. It is bounded
by only implementing the stable core of the handshake, and by pinning the
offered protocol version explicitly so a server-side revision cannot silently
change what we speak.

Deliberately not implemented:

  • **stdio transport.** Each stdio server is a node/python subprocess costing
    50-150 MB against a 1 GB box already running two workers. It also spawns
    processes on behalf of the model, which is exactly what `isolation.py` was
    just added to contain. HTTP servers cost a socket.
  • **Resumability / `Last-Event-Id` replay.** A dropped stream re-issues the
    call instead. Simpler and correct for request-response tools.
  • **Server-initiated requests** (sampling, elicitation, roots). Accepting
    them would let a remote server drive this agent's model, which inverts the
    trust direction; the client advertises no such capabilities.
"""

import os
import json
import time
import logging
import threading

import requests as http_requests

logger = logging.getLogger("my-agent-mini")


# The newest revision reachable through the `initialize` handshake. Pinned
# rather than tracking "latest" because the newest revision overall
# (2026-07-28) drops the handshake entirely for a per-request envelope, which
# this client does not speak. The server answers with its own choice and that
# answer is what gets used; this is only the opening offer.
PROTOCOL_VERSION = "2025-11-25"

CLIENT_INFO = {"name": "my-agent-mini", "version": "1.0.0"}

# A tool call can legitimately take a while (a search, a build query). The
# ceiling matters because the run engine only checks its wall-clock budget
# between steps — a call that hangs forever leaves the run `running` until the
# watchdog reaps it, which costs a whole schedule slot.
DEFAULT_TIMEOUT = 60

# Tool listings are re-fetched at most this often. Servers rarely change their
# tool set mid-session, and the listing is on the path of every `mcp_list`.
TOOL_CACHE_SECONDS = 300


class MCPError(Exception):
    """Any failure talking to an MCP server. Carries a human-readable reason
    because it is rendered straight into a tool result the model reads."""


# ── Configuration ──

def _config_path() -> str:
    return os.environ.get(
        "MCP_CONFIG_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_servers.json"),
    )


def load_server_config() -> dict[str, dict]:
    """Read the server table from `MCP_SERVERS` (inline JSON) or the config file.

    Shape, per server name:

        {"url": "https://…/mcp",
         "headers": {"Authorization": "Bearer …"},
         "tier": "read" | "write_local" | "external",
         "enabled": true}

    The env var wins when both are present, so a deployment can override the
    checked-in file without editing it.
    """
    raw = os.environ.get("MCP_SERVERS", "").strip()
    if not raw:
        path = _config_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as f:
                raw = f.read()
        except OSError as e:
            logger.warning(f"MCP config unreadable at {path}: {e}")
            return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        # Loudly, and return nothing. A malformed config that silently yielded
        # a partial server list would present as "the tool just isn't there".
        logger.error(f"MCP config is not valid JSON, no servers loaded: {e}")
        return {}

    servers = parsed.get("servers", parsed) if isinstance(parsed, dict) else {}
    if not isinstance(servers, dict):
        logger.error("MCP config: expected an object of server name → settings")
        return {}

    return {
        name: cfg for name, cfg in servers.items()
        if isinstance(cfg, dict) and cfg.get("enabled", True)
    }


def configured_server_names() -> list[str]:
    return sorted(load_server_config())


# ── Session ──

class _Session:
    """One initialized connection to one server.

    Not thread-safe on its own; the per-server lock in this module serialises
    access. Two workers sharing one session id is fine sequentially and a
    protocol violation concurrently, and serialising is cheaper than a
    connection pool for a call rate measured in ones per minute.
    """

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.url = (cfg.get("url") or "").strip()
        self.extra_headers = dict(cfg.get("headers") or {})
        self.timeout = int(cfg.get("timeout") or DEFAULT_TIMEOUT)
        self.session_id = ""
        self.protocol_version = ""
        self.server_info: dict = {}
        self._tools: list[dict] = []
        self._tools_fetched_at = 0.0
        self._capabilities: dict = {}
        self._next_id = 0
        self._http = http_requests.Session()

    # ── wire ──

    def _headers(self) -> dict:
        headers = {
            # Both are required: the server chooses which to answer with, and
            # a client that accepts only one gets a 406.
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.protocol_version:
            headers["MCP-Protocol-Version"] = self.protocol_version
        headers.update(self.extra_headers)
        return headers

    def _post(self, payload: dict, expect_reply: bool = True):
        try:
            response = self._http.post(
                self.url,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
        except http_requests.RequestException as e:
            raise MCPError(f"{self.name}: transport error: {str(e)[:200]}") from e

        # The session id is assigned on the initialize response and echoed on
        # every request after it. Servers may also rotate it mid-session.
        new_session = response.headers.get("Mcp-Session-Id")
        if new_session:
            self.session_id = new_session

        if response.status_code == 404 and self.session_id:
            # The server forgot this session. Distinguished from a plain 404 so
            # the caller can re-initialize and retry exactly once.
            raise _SessionLost(f"{self.name}: session expired")
        if response.status_code >= 400:
            raise MCPError(
                f"{self.name}: HTTP {response.status_code}: {response.text[:200]}"
            )

        if not expect_reply:
            return None

        return _decode(response, self.name)

    def _request(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        payload = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            payload["params"] = params

        message = self._post(payload)
        if message is None:
            raise MCPError(f"{self.name}: no response to {method}")
        if "error" in message:
            err = message["error"] or {}
            raise MCPError(
                f"{self.name}: {method} failed: "
                f"{err.get('message', 'unknown error')} (code {err.get('code')})"
            )
        return message.get("result") or {}

    def _notify(self, method: str, params: dict | None = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._post(payload, expect_reply=False)

    # ── lifecycle ──

    def initialize(self) -> None:
        if not self.url:
            raise MCPError(f"{self.name}: no url configured")
        if not self.url.lower().startswith("https://"):
            # Bearer tokens live in these headers. Refuse to put them on the
            # wire in clear text, except against a loopback server where there
            # is no wire.
            host = self.url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
            if host not in ("localhost", "127.0.0.1", "::1"):
                raise MCPError(
                    f"{self.name}: refusing plaintext http:// to a non-local host"
                )

        result = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        })

        # Use the server's counter-offer, not our own request — it is entitled
        # to answer with an older revision, and every later request has to
        # carry the version actually agreed on.
        self.protocol_version = result.get("protocolVersion") or PROTOCOL_VERSION
        self.server_info = result.get("serverInfo") or {}
        self._capabilities = result.get("capabilities") or {}

        self._notify("notifications/initialized")
        logger.info(
            f"MCP connected: {self.name} "
            f"({self.server_info.get('name', '?')} "
            f"{self.server_info.get('version', '?')}, "
            f"protocol {self.protocol_version})"
        )

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

    # ── operations ──

    def list_tools(self, refresh: bool = False) -> list[dict]:
        fresh = time.time() - self._tools_fetched_at < TOOL_CACHE_SECONDS
        if self._tools and fresh and not refresh:
            return self._tools

        # Key presence, not truthiness: `"tools": {}` is how a server with no
        # sub-capabilities advertises tool support, and it is the common case.
        # Testing the value instead would report every such server as empty.
        if "tools" not in (self._capabilities or {}):
            # A server with no tools capability is legal — it may serve only
            # resources or prompts. Empty list, not an error.
            self._tools = []
            self._tools_fetched_at = time.time()
            return self._tools

        tools: list[dict] = []
        cursor = None
        # Bounded, because the page cursor comes from the server and a buggy
        # one that always returns the same cursor would spin here forever.
        for _ in range(20):
            params = {"cursor": cursor} if cursor else {}
            result = self._request("tools/list", params)
            tools.extend(result.get("tools") or [])
            cursor = result.get("nextCursor")
            if not cursor:
                break

        self._tools = tools
        self._tools_fetched_at = time.time()
        return tools

    def call_tool(self, tool: str, arguments: dict) -> str:
        result = self._request("tools/call", {
            "name": tool,
            "arguments": arguments or {},
        })
        text = _render_content(result.get("content") or [])
        if result.get("isError"):
            # A tool-level error is a normal outcome the model should see and
            # reason about, not a transport failure — return it as text.
            return f"⚠️ {self.name}/{tool} reported an error:\n{text}"
        return text or "(tool returned no content)"


class _SessionLost(MCPError):
    """The server no longer recognises our session id; re-initialize and retry."""


# ── Response decoding ──

def _decode(response, server_name: str) -> dict | None:
    """Return the single JSON-RPC message in `response`.

    A server may answer a POST either with `application/json` (one message) or
    with `text/event-stream` (a short SSE stream that ends after the reply).
    Both are spec-legal for the same request, so both are handled here rather
    than assuming whichever one the first server we test against happens to use.
    """
    if response.status_code == 202 or not (response.content or b"").strip():
        # Accepted-with-no-body: the correct answer to a notification.
        return None

    content_type = (response.headers.get("Content-Type") or "").lower()

    if "text/event-stream" in content_type:
        for message in _iter_sse_messages(response.text):
            if "id" in message or "error" in message:
                return message
        return None

    try:
        return response.json()
    except ValueError as e:
        raise MCPError(
            f"{server_name}: response was neither JSON nor SSE "
            f"(content-type {content_type or 'unset'}): {str(e)[:120]}"
        ) from e


def _iter_sse_messages(body: str):
    """Yield the JSON payloads of `data:` lines in an SSE body.

    Written against the wire format rather than pulled from a library: an SSE
    frame is lines until a blank line, `data:` lines concatenate with newlines,
    and a single leading space after the colon is stripped.
    """
    data_lines: list[str] = []
    for line in body.splitlines():
        if not line.strip():
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue
            continue
        if line.startswith(":"):
            continue  # keepalive comment
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))

    if data_lines:
        try:
            yield json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            pass


def _render_content(blocks: list) -> str:
    """Flatten MCP content blocks into text the model can read.

    Non-text blocks are named, not inlined: a base64 image would blow past
    RUN_CONTEXT_LIMIT_CHARS and force a compaction that loses real transcript.
    """
    parts = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text") or "")
        elif kind == "resource":
            resource = block.get("resource") or {}
            if resource.get("text"):
                parts.append(str(resource["text"]))
            else:
                parts.append(f"[resource: {resource.get('uri', 'unknown')}]")
        else:
            parts.append(f"[{kind or 'unknown'} content omitted]")
    return "\n".join(p for p in parts if p).strip()


# ── Session registry ──

_SESSIONS: dict[str, _Session] = {}
_REGISTRY_LOCK = threading.Lock()
# One lock per server, not one global lock. A single lock would mean a 60s call
# against a slow server blocking an unrelated call against a fast one — on a box
# with two workers, that is half the agent stalled on someone else's server.
_SERVER_LOCKS: dict[str, threading.Lock] = {}


def reset() -> None:
    """Drop every cached session. Used by the tests and after a config change."""
    with _REGISTRY_LOCK:
        for session in _SESSIONS.values():
            session.close()
        _SESSIONS.clear()


def _lock_for(name: str) -> threading.Lock:
    with _REGISTRY_LOCK:
        return _SERVER_LOCKS.setdefault(name, threading.Lock())


def _session_for(name: str) -> _Session:
    servers = load_server_config()
    if name not in servers:
        known = ", ".join(sorted(servers)) or "none configured"
        raise MCPError(f"unknown MCP server '{name}' (configured: {known})")

    session = _SESSIONS.get(name)
    if session is None:
        session = _Session(name, servers[name])
        session.initialize()
        _SESSIONS[name] = session
    return session


def _with_session(name: str, operation):
    """Run `operation(session)`, re-initializing once if the session was lost.

    Long-lived servers drop idle sessions, and a scheduled run that fires hours
    after the last one would otherwise fail on its first call every time.
    """
    with _lock_for(name):
        session = _session_for(name)
        try:
            return operation(session)
        except _SessionLost:
            logger.info(f"MCP session lost for {name}, reconnecting")
            session.close()
            _SESSIONS.pop(name, None)
            session = _session_for(name)
            return operation(session)


def list_tools(name: str, refresh: bool = False) -> list[dict]:
    return _with_session(name, lambda s: s.list_tools(refresh=refresh))


def call_tool(name: str, tool: str, arguments: dict) -> str:
    return _with_session(name, lambda s: s.call_tool(tool, arguments))


def server_tier(name: str) -> str:
    """The governor risk tier configured for a server.

    Unset means EXTERNAL, matching `governor.tier_of` for unknown tools: an
    unclassified remote capability is assumed to reach the outside world.
    """
    cfg = load_server_config().get(name) or {}
    tier = str(cfg.get("tier") or "").strip().lower()
    return tier if tier in ("read", "write_local", "external") else "external"
