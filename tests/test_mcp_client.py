"""
MCP client — streamable HTTP transport.

The transport is hand-rolled (see the module docstring for why), so protocol
correctness is ours to own and these tests are where that ownership lives.
They run against a real HTTP server on a real socket rather than a mocked
`requests`, because the things most likely to be wrong are exactly the things
a mock would paper over: header names, the JSON-vs-SSE content-type fork,
session-id echoing, and the notification that carries no reply.

The fake server is deliberately strict — it 400s on a missing header rather
than tolerating it — so a client regression fails here instead of against
somebody's production server.
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mcp_client


# ── A minimal, strict MCP server ──

class _Handler(BaseHTTPRequestHandler):
    behaviour: dict = {}

    def log_message(self, *args):  # keep the test output clean
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        server = self.server
        server.requests.append({"body": body, "headers": dict(self.headers)})

        accept = self.headers.get("Accept", "")
        if "application/json" not in accept or "text/event-stream" not in accept:
            return self._raw(406, "text/plain", b"Not Acceptable")

        method = body.get("method")

        # A notification has no id and must get no JSON-RPC reply.
        if "id" not in body:
            if method == "notifications/initialized":
                server.initialized = True
            return self._raw(202, "text/plain", b"")

        if method != "initialize" and not self.headers.get("Mcp-Session-Id"):
            return self._raw(400, "text/plain", b"missing session id")

        if server.behaviour.get("expire_next") and method != "initialize":
            server.behaviour["expire_next"] = False
            return self._raw(404, "text/plain", b"session expired")

        result = server.results(method, body.get("params") or {})
        if isinstance(result, tuple):  # (error_code, message)
            payload = {
                "jsonrpc": "2.0", "id": body["id"],
                "error": {"code": result[0], "message": result[1]},
            }
        else:
            payload = {"jsonrpc": "2.0", "id": body["id"], "result": result}

        if method == "initialize":
            self.send_response(200)
            self.send_header("Mcp-Session-Id", server.session_id)
        else:
            self.send_response(200)

        if server.behaviour.get("sse"):
            encoded = (
                f": keepalive\n\nevent: message\ndata: {json.dumps(payload)}\n\n"
            ).encode()
            self.send_header("Content-Type", "text/event-stream")
        else:
            encoded = json.dumps(payload).encode()
            self.send_header("Content-Type", "application/json")

        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _raw(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)


def _results(method, params):
    if method == "initialize":
        return {
            "protocolVersion": "2025-06-18",  # deliberately older than we offer
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "0.1"},
        }
    if method == "tools/list":
        if params.get("cursor") == "page2":
            return {"tools": [{"name": "second", "description": "the second one"}]}
        return {
            "tools": [{
                "name": "echo",
                "description": "Echo the input",
                "inputSchema": {"type": "object", "properties": {"text": {}}},
            }],
            "nextCursor": "page2",
        }
    if method == "tools/call":
        name = params.get("name")
        if name == "echo":
            return {"content": [{"type": "text", "text": params["arguments"]["text"]}]}
        if name == "explodes":
            return {"content": [{"type": "text", "text": "disk full"}], "isError": True}
        if name == "image":
            return {"content": [{"type": "image", "data": "A" * 5000}]}
        return (-32602, f"Unknown tool: {name}")
    return (-32601, f"Unknown method: {method}")


@pytest.fixture
def mcp_server(monkeypatch):
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    httpd.requests = []
    httpd.initialized = False
    httpd.session_id = "sess-abc123"
    httpd.behaviour = {}
    httpd.results = _results

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{httpd.server_port}/mcp"
    monkeypatch.setenv("MCP_SERVERS", json.dumps({
        "servers": {"fake": {"url": url, "tier": "read"}}
    }))
    mcp_client.reset()

    yield httpd

    mcp_client.reset()
    httpd.shutdown()
    httpd.server_close()


# ── Handshake ──

def test_initialize_then_list_and_call(mcp_server):
    assert mcp_client.call_tool("fake", "echo", {"text": "hello"}) == "hello"


def test_client_offers_a_pinned_protocol_version(mcp_server):
    mcp_client.list_tools("fake")

    init = mcp_server.requests[0]["body"]
    assert init["method"] == "initialize"
    assert init["params"]["protocolVersion"] == mcp_client.PROTOCOL_VERSION
    assert init["params"]["clientInfo"]["name"] == "my-agent-mini"


def test_client_advertises_no_capabilities(mcp_server):
    """INVARIANT: sampling/elicitation would let a remote server drive this
    agent's model. The trust direction only runs one way."""
    mcp_client.list_tools("fake")

    assert mcp_server.requests[0]["body"]["params"]["capabilities"] == {}


def test_initialized_notification_is_sent_and_expects_no_reply(mcp_server):
    """The handshake is not complete without it; servers are entitled to
    reject everything that follows until it arrives."""
    mcp_client.list_tools("fake")

    assert mcp_server.initialized is True


def test_the_negotiated_version_is_the_servers_answer_not_our_offer(mcp_server):
    """The server may counter-offer an older revision, and every subsequent
    request has to carry the version actually agreed."""
    mcp_client.list_tools("fake")

    later = [r for r in mcp_server.requests if r["body"].get("method") == "tools/list"]
    assert later[0]["headers"]["MCP-Protocol-Version"] == "2025-06-18"


def test_session_id_from_the_handshake_is_echoed_on_later_requests(mcp_server):
    mcp_client.list_tools("fake")

    later = [r for r in mcp_server.requests if r["body"].get("method") == "tools/list"]
    assert later[0]["headers"]["Mcp-Session-Id"] == "sess-abc123"


# ── The JSON / SSE fork ──

@pytest.mark.parametrize("sse", [False, True], ids=["json", "sse"])
def test_both_response_encodings_are_understood(mcp_server, sse):
    """A server may answer the same POST with either application/json or a
    short text/event-stream. Both are spec-legal, so assuming the one our
    first server happened to use would break against the next one."""
    mcp_server.behaviour["sse"] = sse

    assert mcp_client.call_tool("fake", "echo", {"text": "round trip"}) == "round trip"


def test_sse_keepalive_comments_are_skipped(mcp_server):
    mcp_server.behaviour["sse"] = True

    # The fake server prefixes every SSE body with a `: keepalive` comment
    # line; parsing it as data would raise instead of returning.
    assert mcp_client.call_tool("fake", "echo", {"text": "x"}) == "x"


def test_accept_header_offers_both_encodings(mcp_server):
    """The strict fake 406s otherwise — which is what a real server does."""
    mcp_client.list_tools("fake")

    accept = mcp_server.requests[0]["headers"]["Accept"]
    assert "application/json" in accept and "text/event-stream" in accept


# ── Pagination, errors, reconnection ──

def test_tool_listing_follows_the_cursor(mcp_server):
    names = [t["name"] for t in mcp_client.list_tools("fake")]

    assert names == ["echo", "second"]


def test_tool_listing_is_cached(mcp_server):
    mcp_client.list_tools("fake")
    before = sum(1 for r in mcp_server.requests if r["body"].get("method") == "tools/list")
    mcp_client.list_tools("fake")
    after = sum(1 for r in mcp_server.requests if r["body"].get("method") == "tools/list")

    assert before == after


def test_a_jsonrpc_error_becomes_a_readable_message(mcp_server):
    with pytest.raises(mcp_client.MCPError) as excinfo:
        mcp_client.call_tool("fake", "nope", {})

    assert "Unknown tool" in str(excinfo.value)


def test_a_tool_level_error_is_returned_as_text_not_raised(mcp_server):
    """`isError` is the tool saying the work failed, which the model should
    read and reason about. Raising would turn a normal outcome into a
    transport failure and lose the reason."""
    out = mcp_client.call_tool("fake", "explodes", {})

    assert "disk full" in out
    assert "error" in out.lower()


def test_a_lost_session_reconnects_and_retries_once(mcp_server):
    """Servers drop idle sessions. A schedule firing hours later would
    otherwise fail its first call every single time."""
    mcp_client.list_tools("fake")
    mcp_server.behaviour["expire_next"] = True

    assert mcp_client.call_tool("fake", "echo", {"text": "after reconnect"}) == "after reconnect"

    initializes = [r for r in mcp_server.requests if r["body"].get("method") == "initialize"]
    assert len(initializes) == 2


def test_calls_to_different_servers_do_not_block_each_other(mcp_server, monkeypatch):
    """Locking is per server. One global lock would let a slow server stall an
    unrelated call — on a box with two workers, half the agent."""
    url = f"http://127.0.0.1:{mcp_server.server_port}/mcp"
    monkeypatch.setenv("MCP_SERVERS", json.dumps({
        "servers": {"a": {"url": url}, "b": {"url": url}}
    }))
    mcp_client.reset()

    held = threading.Event()
    released = threading.Event()

    def slow(_):
        held.set()
        released.wait(timeout=5)
        return "done"

    slow_thread = threading.Thread(
        target=lambda: mcp_client._with_session("a", slow), daemon=True
    )
    slow_thread.start()
    assert held.wait(timeout=5), "the slow call never started"

    try:
        # Must not block behind server "a".
        assert mcp_client._with_session("b", lambda _: "ok") == "ok"
    finally:
        released.set()
        slow_thread.join(timeout=5)


def test_concurrent_calls_to_one_server_are_serialised(mcp_server):
    """The other half of the same rule: two workers sharing one session id
    concurrently is a protocol violation."""
    mcp_client.list_tools("fake")
    overlaps = []
    active = []
    guard = threading.Lock()

    def observe(_):
        with guard:
            active.append(1)
            overlaps.append(len(active))
        try:
            return "ok"
        finally:
            with guard:
                active.pop()

    threads = [
        threading.Thread(target=lambda: mcp_client._with_session("fake", observe))
        for _ in range(6)
    ]
    [t.start() for t in threads]
    [t.join(timeout=5) for t in threads]

    assert max(overlaps) == 1, f"observed {max(overlaps)} concurrent calls"


def test_non_text_content_is_named_not_inlined(mcp_server):
    """A base64 image would blow past RUN_CONTEXT_LIMIT_CHARS and force a
    compaction that throws away real transcript."""
    out = mcp_client.call_tool("fake", "image", {})

    assert "A" * 100 not in out
    assert "image" in out


def test_an_unreachable_server_reports_rather_than_hangs(monkeypatch):
    monkeypatch.setenv("MCP_SERVERS", json.dumps({
        "servers": {"dead": {"url": "http://127.0.0.1:1/mcp", "timeout": 2}}
    }))
    mcp_client.reset()

    with pytest.raises(mcp_client.MCPError) as excinfo:
        mcp_client.list_tools("dead")

    assert "dead" in str(excinfo.value)


# ── Configuration ──

def test_plaintext_http_to_a_remote_host_is_refused(monkeypatch):
    """INVARIANT: bearer tokens live in these headers. Loopback is exempt
    because there is no wire to sniff."""
    monkeypatch.setenv("MCP_SERVERS", json.dumps({
        "servers": {"insecure": {"url": "http://example.com/mcp"}}
    }))
    mcp_client.reset()

    with pytest.raises(mcp_client.MCPError) as excinfo:
        mcp_client.list_tools("insecure")

    assert "plaintext" in str(excinfo.value)


def test_disabled_servers_are_not_offered(monkeypatch):
    monkeypatch.setenv("MCP_SERVERS", json.dumps({
        "servers": {
            "on": {"url": "https://a/mcp"},
            "off": {"url": "https://b/mcp", "enabled": False},
        }
    }))
    mcp_client.reset()

    assert mcp_client.configured_server_names() == ["on"]


def test_malformed_config_yields_no_servers_rather_than_some(monkeypatch):
    """A partially-parsed config would present as 'the tool just isn't there',
    which is the hardest possible thing to debug from a Slack message."""
    monkeypatch.setenv("MCP_SERVERS", "{not json")
    mcp_client.reset()

    assert mcp_client.configured_server_names() == []


def test_unknown_server_names_the_ones_that_exist(monkeypatch):
    monkeypatch.setenv("MCP_SERVERS", json.dumps({"servers": {"real": {"url": "https://a/mcp"}}}))
    mcp_client.reset()

    with pytest.raises(mcp_client.MCPError) as excinfo:
        mcp_client.list_tools("imaginary")

    assert "real" in str(excinfo.value)


# ── Governor integration ──

def test_an_unclassified_server_is_external(monkeypatch):
    import governor

    monkeypatch.setenv("MCP_SERVERS", json.dumps({"servers": {"plain": {"url": "https://a/mcp"}}}))
    mcp_client.reset()

    assert governor.tier_of("mcp_call", {"server": "plain"}) == governor.EXTERNAL


def test_a_server_classified_read_narrows_the_tier(monkeypatch):
    import governor

    monkeypatch.setenv("MCP_SERVERS", json.dumps({
        "servers": {"docs": {"url": "https://a/mcp", "tier": "read"}}
    }))
    mcp_client.reset()

    assert governor.tier_of("mcp_call", {"server": "docs"}) == governor.READ


def test_tier_without_args_is_the_conservative_answer():
    """INVARIANT: a caller that does not pass args must never get a *weaker*
    tier than one that does."""
    import governor

    assert governor.tier_of("mcp_call") == governor.EXTERNAL
    assert governor.tier_of("mcp_call", {}) == governor.EXTERNAL
    assert governor.tier_of("mcp_call", {"server": ""}) == governor.EXTERNAL


def test_an_unreadable_config_does_not_downgrade_the_tier(monkeypatch):
    import governor

    monkeypatch.setattr(
        mcp_client, "server_tier",
        lambda _: (_ for _ in ()).throw(RuntimeError("config on fire")),
    )

    assert governor.tier_of("mcp_call", {"server": "anything"}) == governor.EXTERNAL


def test_a_bogus_tier_string_is_not_honoured(monkeypatch):
    import governor

    monkeypatch.setenv("MCP_SERVERS", json.dumps({
        "servers": {"sneaky": {"url": "https://a/mcp", "tier": "harmless"}}
    }))
    mcp_client.reset()

    assert governor.tier_of("mcp_call", {"server": "sneaky"}) == governor.EXTERNAL


def test_mcp_call_is_owner_only():
    """The tier narrows per server; the owner axis does not. The set of
    reachable effects is only as bounded as the config, and the config is the
    owner's."""
    import tools

    assert "mcp_call" in tools.OWNER_ONLY_TOOLS


# ── The tools ──

def test_mcp_call_tool_accepts_arguments_as_a_json_string(mcp_server, monkeypatch):
    """The tool protocol is text, so the model sometimes emits the nested
    object as a string instead of an object."""
    import tools

    monkeypatch.setenv("OWNER_SLACK_ID", "U_OWNER")

    assert tools.mcp_call("fake", "echo", '{"text": "as a string"}') == "as a string"


def test_mcp_call_tool_rejects_a_non_object_arguments(mcp_server, monkeypatch):
    import tools

    monkeypatch.setenv("OWNER_SLACK_ID", "U_OWNER")

    assert "must be a JSON object" in tools.mcp_call("fake", "echo", 42)


def test_mcp_list_without_a_server_lists_servers_and_tiers(mcp_server):
    import tools

    out = tools.mcp_list()

    assert "fake" in out and "read" in out


def test_mcp_list_with_a_server_shows_tools_and_arguments(mcp_server):
    import tools

    out = tools.mcp_list("fake")

    assert "echo" in out and "text" in out


def test_mcp_list_says_so_when_nothing_is_configured(monkeypatch):
    import tools

    monkeypatch.setenv("MCP_SERVERS", json.dumps({"servers": {}}))
    mcp_client.reset()

    assert "No MCP servers are configured" in tools.mcp_list()


def test_tool_errors_reach_the_model_as_text_not_exceptions(monkeypatch):
    """`run_tool` catches exceptions, but a raised MCPError would arrive as a
    generic traceback string instead of the reason."""
    import tools

    monkeypatch.setenv("MCP_SERVERS", json.dumps({"servers": {}}))
    mcp_client.reset()

    assert tools.mcp_call("gone", "x", {}).startswith("❌")
