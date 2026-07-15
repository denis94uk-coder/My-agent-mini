"""
Concept Graph — a lightweight entity-relationship layer on top of SQLite memory.

Extracts durable entities (projects, tools, decisions, people, technologies)
and relationships from conversations, persists them in SQLite, and provides
graph-walking recall that complements the existing FTS5 keyword search.

Design constraints (1 GB Oracle Cloud VM):
  - No embeddings, no extra ML models — extraction is rule-based + LLM-assisted
  - SQLite-backed (same memory.db), not a separate service
  - NetworkX used only for in-memory traversal (lazy-loaded, rebuilt from SQLite)
  - ~2-5 MB RAM overhead for a graph with hundreds of nodes

Usage:
  concept_graph.extract_and_store(text, conv_key)   — after each message
  concept_graph.recall(query, limit=5)              — graph-enhanced recall
  concept_graph.get_entity_context(entity_name)     — deep dive on one entity
"""

import re
import time
import json
import sqlite3
import logging
import threading
from pathlib import Path

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

logger = logging.getLogger("my-agent-mini")

DB_PATH = Path.home() / "my-agent-mini" / "memory.db"

# ── Schema ──

_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    entity_type TEXT NOT NULL DEFAULT 'concept',
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_entity_name ON graph_entities(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_entity_type ON graph_entities(entity_type);

CREATE TABLE IF NOT EXISTS graph_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES graph_entities(id),
    target_id INTEGER NOT NULL REFERENCES graph_entities(id),
    relation TEXT NOT NULL DEFAULT 'related_to',
    weight REAL NOT NULL DEFAULT 1.0,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    conv_key TEXT,
    snippet TEXT,
    UNIQUE(source_id, target_id, relation)
);
CREATE INDEX IF NOT EXISTS idx_edge_source ON graph_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edge_target ON graph_edges(target_id);
"""

_lock = threading.Lock()
_graph_cache: "nx.DiGraph | None" = None
_graph_cache_ts: float = 0.0
CACHE_TTL = 300  # rebuild in-memory graph every 5 min


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


# ── Entity types and extraction patterns ──

# Fast rule-based extraction — no LLM call needed per message.
# The LLM-assisted extraction runs asynchronously after thread summaries.

ENTITY_TYPES = {
    "project":    r"\b(?:repo|project|app|bot|service|codebase)\b",
    "technology": r"\b(?:python|javascript|react|nextjs|node|sqlite|networkx|"
                  r"scrapling|vercel|supabase|docker|systemd|nginx|redis|"
                  r"postgresql|mongodb|fastapi|flask|django|slack|github|"
                  r"openai|gemini|groq|nvidia|pollinations|ruff|pytest|"
                  r"fts5|bm25|oracle|aws|gcp|azure|html|css|tailwind)\b",
    "tool":       r"\b(?:web_search|fetch_url|run_shell|run_python|write_file|"
                  r"read_file|remember|memory_search|create_plan|update_task|"
                  r"clone_repo|push_branch|repo_edit_file|repo_check|"
                  r"get_weather|github_write_file|github_read_file)\b",
    "person":     r"(?:@[\w.-]+|<@U[A-Z0-9]+>)",
    "decision":   r"(?:decided|decision|agreed|parked|deferred|priority|"
                  r"must not|don't build|do not build|won't|will not)\b",
}

# Known project names — bootstrapped, extended as the graph grows.
_PROJECT_NAMES = {
    "my-agent-mini", "my agent mini", "my agent", "agent mini",
}

# Relationship extraction patterns
_RELATION_PATTERNS = [
    (r"(\w+)\s+(?:uses?|using|built with|powered by|depends on)\s+(\w+)", "uses"),
    (r"(\w+)\s+(?:replaces?|replacing|replaced by)\s+(\w+)", "replaces"),
    (r"(\w+)\s+(?:extends?|extending|enhances?|enhancing)\s+(\w+)", "extends"),
    (r"(\w+)\s+(?:needs?|requires?|requiring)\s+(\w+)", "requires"),
    (r"(?:parked|deferred|postponed)\s+(\w+)", "parked"),
    (r"(\w+)\s+(?:deployed|shipped|merged|landed)", "shipped"),
]


def _extract_entities(text: str) -> list[tuple[str, str]]:
    """Extract (name, type) pairs from a message using rules."""
    entities = []
    text_lower = text.lower()

    # Technology mentions
    for m in re.finditer(ENTITY_TYPES["technology"], text_lower):
        name = m.group(0)
        entities.append((name, "technology"))

    # Tool mentions
    for m in re.finditer(ENTITY_TYPES["tool"], text_lower):
        entities.append((m.group(0), "tool"))

    # Person mentions (@user or <@UXXXXX>)
    for m in re.finditer(ENTITY_TYPES["person"], text):
        entities.append((m.group(0), "person"))

    # Known project names
    for proj in _PROJECT_NAMES:
        if proj in text_lower:
            entities.append((proj, "project"))

    # Decision-related phrases — extract the surrounding context as entity
    for m in re.finditer(ENTITY_TYPES["decision"], text_lower):
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 60)
        snippet = text[start:end].strip()
        # Normalize to a short key
        key = re.sub(r"[^a-z0-9 ]", "", snippet.lower())[:60].strip()
        if len(key) > 10:
            entities.append((key, "decision"))

    return entities


def _extract_relations(text: str, entities: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Extract (source, target, relation) triples."""
    relations = []
    entity_names = {e[0].lower() for e in entities}

    for pattern, rel_type in _RELATION_PATTERNS:
        for m in re.finditer(pattern, text.lower()):
            groups = m.groups()
            if len(groups) >= 2:
                src, tgt = groups[0], groups[1]
                if src in entity_names or tgt in entity_names:
                    relations.append((src, tgt, rel_type))

    # Co-occurrence: entities mentioned in the same message are related
    ent_list = [(name, etype) for name, etype in entities
                if etype in ("technology", "project", "tool")]
    for i in range(len(ent_list)):
        for j in range(i + 1, min(i + 4, len(ent_list))):
            relations.append((ent_list[i][0], ent_list[j][0], "co_mentioned"))

    return relations


# ── Storage ──

def _upsert_entity(conn: sqlite3.Connection, name: str, entity_type: str) -> int:
    """Insert or update an entity, return its id."""
    now = time.time()
    row = conn.execute(
        "SELECT id FROM graph_entities WHERE name = ? COLLATE NOCASE",
        (name,),
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE graph_entities SET last_seen = ?, mention_count = mention_count + 1 "
            "WHERE id = ?",
            (now, row[0]),
        )
        return row[0]
    else:
        cur = conn.execute(
            "INSERT INTO graph_entities (name, entity_type, first_seen, last_seen) "
            "VALUES (?, ?, ?, ?)",
            (name, entity_type, now, now),
        )
        return cur.lastrowid


def _upsert_edge(conn: sqlite3.Connection, source_id: int, target_id: int,
                 relation: str, conv_key: str = "", snippet: str = ""):
    """Insert or strengthen an edge."""
    now = time.time()
    row = conn.execute(
        "SELECT id, weight FROM graph_edges "
        "WHERE source_id = ? AND target_id = ? AND relation = ?",
        (source_id, target_id, relation),
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE graph_edges SET weight = ?, last_seen = ?, snippet = ? WHERE id = ?",
            (row[1] + 1.0, now, snippet[:300] if snippet else "", row[0]),
        )
    else:
        conn.execute(
            "INSERT INTO graph_edges (source_id, target_id, relation, weight, "
            "first_seen, last_seen, conv_key, snippet) VALUES (?, ?, ?, 1.0, ?, ?, ?, ?)",
            (source_id, target_id, relation, now, now, conv_key, snippet[:300]),
        )


def extract_and_store(text: str, conv_key: str = ""):
    """
    Extract entities and relationships from text and persist to SQLite.
    Designed to be called after each message — fast enough for inline use.
    """
    if not text or len(text) < 10:
        return

    entities = _extract_entities(text)
    if not entities:
        return

    relations = _extract_relations(text, entities)

    with _lock:
        conn = _get_db()
        try:
            # Upsert entities
            id_map = {}
            for name, etype in entities:
                eid = _upsert_entity(conn, name, etype)
                id_map[name.lower()] = eid

            # Upsert edges
            for src, tgt, rel in relations:
                src_id = id_map.get(src.lower())
                tgt_id = id_map.get(tgt.lower())
                if src_id and tgt_id and src_id != tgt_id:
                    _upsert_edge(conn, src_id, tgt_id, rel, conv_key, text[:300])

            conn.commit()
        except Exception as e:
            logger.warning(f"Concept graph extraction failed: {e}")
        finally:
            conn.close()

    # Invalidate cache
    global _graph_cache_ts
    _graph_cache_ts = 0


# ── LLM-assisted extraction (called from thread summary, not per-message) ──

def extract_from_summary_async(summary: str, conv_key: str, call_ai_fn=None):
    """
    Deeper extraction from a thread summary using the LLM. Called in the
    background after _maybe_summarize_thread, so it never delays replies.
    If call_ai_fn is None, falls back to rule-based only.
    """
    def _worker():
        try:
            # Always do rule-based extraction
            extract_and_store(summary, conv_key)

            if call_ai_fn is None:
                return

            prompt = (
                "Extract entities and relationships from this conversation summary.\n"
                "Return ONLY valid JSON, no markdown fences, no explanation:\n"
                '{"entities": [{"name": "...", "type": "project|technology|person|decision|concept"}], '
                '"relationships": [{"source": "...", "target": "...", "relation": "uses|requires|replaces|extends|parked|shipped|related_to"}]}\n\n'
                f"Summary:\n{summary[:1500]}"
            )
            result = call_ai_fn(
                [{"role": "user", "content": prompt}],
                "You extract structured knowledge from text. Return only JSON.",
            )
            if not result or result.startswith("❌"):
                return

            # Parse JSON from response (handle markdown fences)
            json_match = re.search(r"\{.*\}", result, re.DOTALL)
            if not json_match:
                return
            data = json.loads(json_match.group())

            with _lock:
                conn = _get_db()
                try:
                    id_map = {}
                    for ent in data.get("entities", []):
                        name = ent.get("name", "").strip().lower()
                        etype = ent.get("type", "concept").strip().lower()
                        if name and len(name) > 1:
                            eid = _upsert_entity(conn, name, etype)
                            id_map[name] = eid

                    for rel in data.get("relationships", []):
                        src = rel.get("source", "").strip().lower()
                        tgt = rel.get("target", "").strip().lower()
                        rtype = rel.get("relation", "related_to").strip().lower()
                        if src in id_map and tgt in id_map and id_map[src] != id_map[tgt]:
                            _upsert_edge(conn, id_map[src], id_map[tgt], rtype, conv_key)

                    conn.commit()
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"LLM graph extraction parse error: {e}")
                finally:
                    conn.close()

        except Exception as e:
            logger.warning(f"Async graph extraction failed: {e}")

    threading.Thread(target=_worker, daemon=True).start()


# ── In-memory graph (NetworkX) ──

def _build_graph() -> "nx.DiGraph":
    """Rebuild the NetworkX graph from SQLite."""
    global _graph_cache, _graph_cache_ts

    if not NX_AVAILABLE:
        return None

    now = time.time()
    if _graph_cache and (now - _graph_cache_ts) < CACHE_TTL:
        return _graph_cache

    G = nx.DiGraph()
    conn = _get_db()
    try:
        entities = conn.execute(
            "SELECT id, name, entity_type, mention_count, last_seen FROM graph_entities"
        ).fetchall()
        for eid, name, etype, count, last_seen in entities:
            G.add_node(eid, name=name, type=etype, mentions=count, last_seen=last_seen)

        edges = conn.execute(
            "SELECT source_id, target_id, relation, weight, snippet FROM graph_edges"
        ).fetchall()
        for src, tgt, rel, weight, snippet in edges:
            if G.has_node(src) and G.has_node(tgt):
                G.add_edge(src, tgt, relation=rel, weight=weight, snippet=snippet or "")
    finally:
        conn.close()

    with _lock:
        _graph_cache = G
        _graph_cache_ts = now
    return G


def _find_entity_ids(query: str) -> list[int]:
    """Find entity IDs matching query terms."""
    conn = _get_db()
    try:
        # Exact match first
        row = conn.execute(
            "SELECT id FROM graph_entities WHERE name = ? COLLATE NOCASE",
            (query.strip(),),
        ).fetchone()
        if row:
            return [row[0]]

        # Keyword search across entity names
        terms = [w for w in re.findall(r"[a-z0-9_-]{3,}", query.lower())
                 if w not in _STOPWORDS]
        ids = []
        for term in terms[:8]:
            rows = conn.execute(
                "SELECT id FROM graph_entities WHERE name LIKE ? COLLATE NOCASE "
                "ORDER BY mention_count DESC LIMIT 5",
                (f"%{term}%",),
            ).fetchall()
            ids.extend(r[0] for r in rows)
        return list(dict.fromkeys(ids))  # dedupe, preserve order
    finally:
        conn.close()


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of",
    "and", "or", "in", "on", "at", "for", "with", "this", "that", "it",
    "i", "you", "he", "she", "we", "they", "my", "your", "me", "do", "does",
    "did", "can", "could", "would", "should", "will", "what", "how", "why",
    "about", "tell", "show", "get", "just", "now", "please", "know",
}


# ── Public API ──

def recall(query: str, limit: int = 5) -> list[dict]:
    """
    Graph-enhanced recall: find entities matching the query, then walk their
    neighborhood to surface related concepts, decisions, and context.
    Returns a list of dicts with 'entity', 'type', 'relation', 'connected_to',
    'snippet', 'mentions'.
    """
    if not NX_AVAILABLE:
        return []

    G = _build_graph()
    if G is None or len(G) == 0:
        return []

    seed_ids = _find_entity_ids(query)
    if not seed_ids:
        return []

    results = []
    seen = set()

    for seed_id in seed_ids[:4]:
        if seed_id not in G:
            continue
        node = G.nodes[seed_id]
        # Add the seed entity itself
        if seed_id not in seen:
            seen.add(seed_id)
            results.append({
                "entity": node["name"],
                "type": node["type"],
                "relation": "direct_match",
                "connected_to": "",
                "snippet": "",
                "mentions": node["mentions"],
            })

        # Walk 1-hop neighbors (both directions for a DiGraph)
        neighbors = set(G.successors(seed_id)) | set(G.predecessors(seed_id))
        scored = []
        for nid in neighbors:
            if nid in seen:
                continue
            nnode = G.nodes[nid]
            # Score: edge weight * recency * mention count
            edge_data = G.get_edge_data(seed_id, nid) or G.get_edge_data(nid, seed_id) or {}
            weight = edge_data.get("weight", 1.0)
            age_days = max(1.0, (time.time() - nnode["last_seen"]) / 86400)
            score = weight * nnode["mentions"] / (1 + age_days * 0.1)
            scored.append((score, nid, nnode, edge_data))

        scored.sort(key=lambda x: x[0], reverse=True)
        for score, nid, nnode, edge_data in scored[:6]:
            if nid not in seen:
                seen.add(nid)
                results.append({
                    "entity": nnode["name"],
                    "type": nnode["type"],
                    "relation": edge_data.get("relation", "related_to"),
                    "connected_to": node["name"],
                    "snippet": edge_data.get("snippet", "")[:200],
                    "mentions": nnode["mentions"],
                })

    return results[:limit]


def get_entity_context(entity_name: str) -> dict:
    """
    Deep dive on a single entity: its type, all relationships, mention count,
    and recent conversation snippets.
    """
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT id, name, entity_type, mention_count, first_seen, last_seen, metadata "
            "FROM graph_entities WHERE name = ? COLLATE NOCASE",
            (entity_name.strip(),),
        ).fetchone()
        if not row:
            return {"error": f"Entity '{entity_name}' not found in concept graph"}

        eid, name, etype, mentions, first, last, meta = row

        edges_out = conn.execute(
            "SELECT e.name, ge.relation, ge.weight, ge.snippet "
            "FROM graph_edges ge JOIN graph_entities e ON e.id = ge.target_id "
            "WHERE ge.source_id = ? ORDER BY ge.weight DESC LIMIT 20",
            (eid,),
        ).fetchall()

        edges_in = conn.execute(
            "SELECT e.name, ge.relation, ge.weight, ge.snippet "
            "FROM graph_edges ge JOIN graph_entities e ON e.id = ge.source_id "
            "WHERE ge.target_id = ? ORDER BY ge.weight DESC LIMIT 20",
            (eid,),
        ).fetchall()

        return {
            "name": name,
            "type": etype,
            "mentions": mentions,
            "first_seen": time.strftime("%Y-%m-%d", time.localtime(first)),
            "last_seen": time.strftime("%Y-%m-%d", time.localtime(last)),
            "outgoing": [
                {"target": e[0], "relation": e[1], "weight": e[2], "snippet": e[3][:150]}
                for e in edges_out
            ],
            "incoming": [
                {"source": e[0], "relation": e[1], "weight": e[2], "snippet": e[3][:150]}
                for e in edges_in
            ],
        }
    finally:
        conn.close()


def get_graph_stats() -> dict:
    """Summary stats for /botstatus and /health."""
    conn = _get_db()
    try:
        entities = conn.execute("SELECT COUNT(*) FROM graph_entities").fetchone()[0]
        edges = conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
        types = conn.execute(
            "SELECT entity_type, COUNT(*) FROM graph_entities GROUP BY entity_type"
        ).fetchall()
        return {
            "entities": entities,
            "edges": edges,
            "types": {t: c for t, c in types},
            "networkx_available": NX_AVAILABLE,
        }
    except sqlite3.OperationalError:
        return {"entities": 0, "edges": 0, "types": {}, "networkx_available": NX_AVAILABLE}
    finally:
        conn.close()


def format_recall_for_prompt(results: list[dict]) -> str:
    """Format graph recall results as context for the agent prompt."""
    if not results:
        return ""
    lines = ["[CONCEPT GRAPH — related entities and connections from your knowledge base]"]
    for r in results:
        conn_str = f" ({r['relation']} → {r['connected_to']})" if r["connected_to"] else ""
        lines.append(f"  • {r['entity']} [{r['type']}]{conn_str} — {r['mentions']} mentions")
        if r.get("snippet"):
            lines.append(f"    context: {r['snippet'][:150]}")
    return "\n".join(lines)
