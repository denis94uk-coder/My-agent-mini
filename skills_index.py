"""
Retrieval over the vendored skill playbooks in `skills/`.

The library has been in the repo but out of reach. `agent.py`'s prompt tells
the model that "full references live in `skills/coding-practices/` — read the
relevant file when a task warrants it", while `read_file` basenames its
argument into `~/agent_workspace` and `repo_read_file` resolves under the
clone directory. Neither can open the bot's own checkout, which the skills
README says plainly: the files are there "for a future tool that can read this
repo directly". This is that tool.

Retrieval rather than prompt text, for the reason the prompt is already short:
24 playbooks are ~289 KB, and the system prompt is re-sent on every call in a
loop that fires up to MAX_ITERATIONS times. Inlining even the index costs that
on every step of every turn; a search tool costs one call, only when a task
actually warrants one.

Scoring is deterministic term overlap — no AI call and no embeddings, which
stays on the right side of the same decision that deferred embeddings for
memory. FTS5 is not used either: that index is built for `memory.db` rows,
while these are 24 files on disk that parse in milliseconds.
"""

import os
import re
import math
import functools
import logging

logger = logging.getLogger("my-agent-mini")

SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")

# How much of a playbook one tool result may carry. `run_tool` truncates at
# 4000 anyway; stopping here instead means the cut lands on a heading boundary
# with a note, rather than mid-sentence.
MAX_BODY_CHARS = 3500

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9+#.-]{1,}")

# Words that match everything in a corpus about software engineering, and so
# rank nothing. Without this, "how do I test the code" scores every playbook
# equally on "code" and the ranking is noise.
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "you", "your", "are", "not",
    "use", "using", "used", "when", "how", "what", "why", "who", "should",
    "can", "will", "from", "into", "any", "all", "its", "have", "has", "was",
    "code", "codebase", "file", "files", "work", "make", "need", "want",
    "skill", "skills", "agent", "please", "help", "about",
    # Speech verbs and pronouns. "tell" is the one that actually bit: it
    # appears in the observability description ("tells you…") and was enough
    # to make "tell me a joke about penguins" a match.
    "tell", "tells", "told", "say", "says", "said", "give", "gives", "get",
    "let", "lets", "does", "did", "here", "there", "then", "than", "some",
}


def _terms(text: str) -> list[str]:
    return [
        w.lower() for w in _WORD.findall(text or "")
        if w.lower() not in _STOPWORDS and len(w) > 2
    ]


@functools.lru_cache(maxsize=1)
def _document_frequency() -> dict[str, int]:
    """How many playbooks each term appears in, over name + description.

    Computed from the corpus rather than curated, because a hand-written
    stoplist cannot anticipate which words happen to be generic across *these*
    24 documents ("time", "design", "quality") and which are decisive ("spec",
    "attackers", "flaky").
    """
    frequency: dict[str, int] = {}
    for skill in _load():
        seen = set(_terms(skill.name.replace("-", " "))) | set(_terms(skill.description))
        for term in seen:
            frequency[term] = frequency.get(term, 0) + 1
    return frequency


def _idf(term: str) -> float:
    """Inverse document frequency, clamped to [0, 1].

    A term in one playbook scores ~1.0; a term in half of them scores ~0.3; a
    term in nearly all of them scores ~0. Unknown terms get 1.0 — a word absent
    from every description is maximally specific when it does hit the body.
    """
    total = len(_load()) or 1
    appearances = _document_frequency().get(term, 0)
    if appearances <= 0:
        return 1.0
    return max(0.0, min(1.0, math.log(total / appearances) / math.log(total)))


class Skill:
    __slots__ = ("name", "description", "path", "headings", "body")

    def __init__(self, name, description, path, headings, body):
        self.name = name
        self.description = description
        self.path = path
        self.headings = headings
        self.body = body

    def score(self, query_terms: list[str]) -> float:
        """Weighted term overlap.

        The frontmatter `description` is the strongest signal by design: it is
        written as "use when X", which is the question being asked. The body is
        weighted lowest because a long playbook mentions many things once and
        would otherwise beat a short, exactly-relevant one on volume.

        A hit in the name or description is *required*. Scoring on body and
        heading terms alone let "tell me a joke about penguins" match the
        observability playbook, because a 300-line document about software
        contains almost any software-adjacent word somewhere. Body matches
        break ties between real candidates; they do not create one.
        """
        if not query_terms:
            return 0.0

        name_terms = set(_terms(self.name.replace("-", " ")))
        description_terms = set(_terms(self.description))
        heading_terms = set(_terms(" ".join(self.headings)))
        body_terms = set(_terms(self.body))

        total = 0.0
        strong_hit = False
        matched = 0
        for term in query_terms:
            # Rare terms carry the meaning. "spec" or "attackers" appear in one
            # or two playbooks and identify them; "time" or "design" appear in
            # most and identify nothing, which is how "what time is my meeting"
            # matched two playbooks on a single incidental word. A hand-written
            # stoplist cannot keep up with this — the weighting has to come
            # from the corpus.
            weight = _idf(term)
            if term in name_terms:
                # Weighted above a description hit, and deliberately *not*
                # IDF-discounted like every other field. A name is an identity,
                # not a mention: the playbook called spec-driven-development is
                # the answer for "spec" however many other descriptions happen
                # to use the word. With either of those wrong,
                # planning-and-task-breakdown outranked it for "write a spec
                # before I start coding".
                total += 6.0
                strong_hit = True
            elif term in description_terms:
                total += 3.0 * weight
                strong_hit = True
            elif term in heading_terms:
                total += 1.5 * weight
            elif term in body_terms:
                total += 0.5 * weight
            else:
                continue
            matched += 1

        if not strong_hit:
            return 0.0

        # One word in common is coincidence; two is a topic. "what time is my
        # meeting" overlapped exactly one description term and scored above
        # every real query — IDF cannot help there, because "meeting" genuinely
        # is a rare word that genuinely does appear in one description. Every
        # real query in the tests overlaps at least two.
        if len(query_terms) > 1 and matched < 2:
            return 0.0
        # Normalise by query length so a long question is not automatically a
        # higher score than a short one — the threshold has to mean the same
        # thing for both.
        return total / len(query_terms)


def _parse(path: str) -> Skill | None:
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        logger.warning(f"Skill unreadable at {path}: {e}")
        return None

    name, description, body = "", "", text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            frontmatter, body = text[3:end], text[end + 4:]
            for line in frontmatter.splitlines():
                key, _, value = line.partition(":")
                key, value = key.strip().lower(), value.strip()
                if key == "name":
                    name = value
                elif key == "description":
                    description = value

    if not name:
        # Fall back to the directory name, which is how these are referred to
        # anyway ("test-driven-development").
        name = os.path.basename(os.path.dirname(path))

    headings = re.findall(r"^#+\s*(.+)$", body, re.MULTILINE)
    return Skill(name, description, path, headings, body.strip())


@functools.lru_cache(maxsize=1)
def _load() -> list[Skill]:
    """Parse every SKILL.md under `skills/`. Cached — these are static files
    shipped with the code, and re-reading 289 KB per search is pure waste."""
    if not os.path.isdir(SKILLS_DIR):
        return []

    skills = []
    for root, _dirs, files in os.walk(SKILLS_DIR):
        for filename in files:
            if filename.upper() == "SKILL.MD":
                parsed = _parse(os.path.join(root, filename))
                if parsed:
                    skills.append(parsed)
    return sorted(skills, key=lambda s: s.name)


def reload() -> None:
    _load.cache_clear()


def all_skills() -> list[Skill]:
    return _load()


def search(query: str, limit: int = 4, threshold: float = 0.75) -> list[Skill]:
    """Playbooks matching `query`, best first.

    The threshold is what keeps this honest: a question no playbook covers
    returns nothing rather than the least-bad match. Handing the model an
    irrelevant playbook is worse than handing it none — it reads as
    instruction, not as a suggestion it is free to ignore.
    """
    query_terms = _terms(query)
    if not query_terms:
        return []

    scored = [(s.score(query_terms), s) for s in _load()]
    return [s for score, s in sorted(scored, key=lambda p: -p[0]) if score >= threshold][:limit]


def get(name: str) -> Skill | None:
    """One playbook by name, matched loosely — the model will write
    "test driven development" for `test-driven-development`."""
    wanted = re.sub(r"[^a-z0-9]", "", (name or "").lower())
    if not wanted:
        return None
    for skill in _load():
        if re.sub(r"[^a-z0-9]", "", skill.name.lower()) == wanted:
            return skill
    for skill in _load():
        if wanted in re.sub(r"[^a-z0-9]", "", skill.name.lower()):
            return skill
    return None


def render(skill: Skill, max_chars: int = MAX_BODY_CHARS) -> str:
    """A playbook as tool-result text, cut on a heading boundary when long."""
    body = skill.body
    if len(body) > max_chars:
        cut = body.rfind("\n#", 0, max_chars)
        if cut < max_chars // 2:
            cut = max_chars
        body = body[:cut].rstrip() + (
            f"\n\n…(truncated — {len(skill.body) - cut} more characters in "
            f"{os.path.relpath(skill.path, SKILLS_DIR)})"
        )
    return f"# {skill.name}\n\n{body}"


def index_lines() -> list[str]:
    """One line per playbook, for the no-query listing."""
    return [
        f"  • {s.name} — {(s.description or '(no description)')[:160]}"
        for s in _load()
    ]
