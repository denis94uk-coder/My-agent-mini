"""
Critic gate — a second pass that decides whether "done" is actually done.

The agent grades its own homework: it stops when it *feels* finished, which
is exactly when this codebase has failed before (see `_INTENT_ONLY_PATTERNS`
in agent.py — the bot announcing it saved something it never saved). The
narration nudge catches the crude version of that inside a single step. This
catches the version that survives to the end of a whole task: a final answer
that claims work the transcript doesn't support, or covers two thirds of the
goal and calls it shipped.

It generalizes what `tools._run_quality_gate` already does for code — run an
independent check before accepting, and refuse to ship what fails it — to
tasks that have no test suite to run.

Shape: after a final answer, a separate AI call re-reads the goal, the tool
transcript, and the proposed answer, and returns ACCEPT or REVISE + a
specific reason. REVISE pushes the reason back into the loop as another turn.
Capped rounds, and on the last round the work ships anyway with the
unresolved critique attached — a stuck critic must never eat the result.

Two deliberate failure directions:
  • **Fail open.** An unparseable or erroring critic ACCEPTS (logged). A
    broken grader must not become a broken agent.
  • **Blocked ≠ incomplete.** An unattended run that correctly stopped at a
    blocked deploy and said so is *finished*. Without that rule the critic
    demands the impossible thing every round until the cap.
"""

import os
import re
import logging

logger = logging.getLogger("my-agent-mini")

ACCEPT = "accept"
REVISE = "revise"


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    return os.getenv(name, "true" if default else "false").strip().lower() in (
        "1", "true", "yes", "on",
    )


def enabled() -> bool:
    """Whether the gate runs on background/unattended runs."""
    return _bool_env("CRITIC_ENABLED", True)


def interactive_enabled() -> bool:
    """
    Whether the gate also runs on live Slack replies.

    Off by default: it costs an extra AI call per turn, and interactive work
    has a human reading it who can just say "no, you missed X". Unattended
    runs have nobody, which is where the gate earns its cost.
    """
    return _bool_env("CRITIC_INTERACTIVE", False)


def max_rounds() -> int:
    return max(0, _int_env("CRITIC_MAX_ROUNDS", 2))


class Verdict:
    """ACCEPT, or REVISE with a specific, actionable reason."""

    __slots__ = ("kind", "raw", "reason")

    def __init__(self, kind: str, reason: str = "", raw: str = ""):
        self.kind = kind
        self.reason = reason
        self.raw = raw

    @property
    def accepted(self) -> bool:
        return self.kind == ACCEPT

    def __repr__(self):
        return f"Verdict({self.kind}, {self.reason[:60]!r})"


CRITIC_PROMPT = """You are a critic reviewing whether an AI agent actually \
finished the task it was given. You are not the agent and you do not continue \
its work — you deliver one verdict.

THE GOAL THE AGENT WAS GIVEN:
{goal}

WHAT THE AGENT ACTUALLY DID (tool calls and their real results, in order):
{transcript}

THE ANSWER THE AGENT WANTS TO DELIVER:
{final}

Decide: is this genuinely done, or not?

REVISE it when:
- The answer claims an action the transcript does not show actually happening
  ("I saved it", "I've updated the file") with no matching tool result. This is
  the single most important thing you are checking for.
- Part of the goal was silently dropped — a multi-part request where only some
  parts were done, with no mention of the rest.
- The answer states a fact, number, or result that contradicts what the tool
  results actually returned.
- A tool failed and the agent moved on as if it had succeeded.

ACCEPT it when:
- The goal is met, even if the answer is terse or imperfectly worded.
- The agent could not complete something and says so honestly — a blocked
  tool, a missing credential or token, a permission refusal, something needing
  a human decision. Reporting a real blocker IS a complete outcome. Never
  demand work the agent was structurally prevented from doing; it will just
  fail the same way again.
- The remaining gap is style, tone, formatting, or thoroughness you would have
  preferred. You are checking whether it is done, not whether it is elegant.

Answer in exactly this format, nothing else:

VERDICT: ACCEPT
or
VERDICT: REVISE
REASON: <one or two sentences, specific and actionable — name the exact thing \
that is missing or unsupported, and what the agent must do about it. Never \
vague ("could be better"), never a request for polish.>"""


REVISION_TEMPLATE = """[CRITIC — this work was reviewed and is not finished]

{reason}

Fix exactly this now. If it needs a tool, call it in this response — do not \
describe what you would do. If the critic is wrong because you genuinely \
cannot do it (a tool is blocked, a token is missing, it needs a human \
decision), say so plainly and give your final answer; that is a valid \
outcome. Do not restart work you already completed."""


def format_transcript(steps: list[dict], limit: int = 12, result_chars: int = 500) -> str:
    """
    Render the tool trail for the critic.

    Only tool calls and their real results — this is the evidence the critic
    checks the final answer against, so the agent's own prose is deliberately
    left out. It cannot vouch for itself.
    """
    if not steps:
        return "(no tools were used — the agent answered directly)"

    shown = steps[-limit:]
    lines = []
    if len(steps) > limit:
        lines.append(f"... ({len(steps) - limit} earlier steps omitted)")
    for i, step in enumerate(shown, start=len(steps) - len(shown) + 1):
        result = (step.get("result") or "").strip().replace("\n", " ")
        if len(result) > result_chars:
            result = result[:result_chars] + " …(truncated)"
        lines.append(f"{i}. {step.get('tool', '?')} → {result or '(empty result)'}")
    return "\n".join(lines)


def parse_verdict(text: str) -> Verdict:
    """
    Read the critic's reply.

    Anything unparseable is an ACCEPT: a grader that has gone off the rails
    must not be able to trap finished work in a revision loop.
    """
    raw = (text or "").strip()
    match = re.search(r"VERDICT:\s*(ACCEPT|REVISE)", raw, re.IGNORECASE)
    if not match:
        logger.warning(f"Critic gave an unparseable verdict, accepting: {raw[:200]}")
        return Verdict(ACCEPT, raw=raw)

    if match.group(1).upper() == "ACCEPT":
        return Verdict(ACCEPT, raw=raw)

    reason_match = re.search(r"REASON:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else ""
    if not reason:
        logger.warning("Critic said REVISE with no reason, accepting instead")
        return Verdict(ACCEPT, raw=raw)
    return Verdict(REVISE, reason=reason[:1500], raw=raw)


# A refusal the agent could not have worked around. The prompt asks the critic
# to accept these; that only holds while the critic model is strong enough to
# follow it. The transcript already carries the evidence in a fixed form, so
# the check belongs in code as well — a weak critic route must not be able to
# demand impossible work every round until the cap.
BLOCKED_MARKERS = (
    "not authorized:",
    "denied: a human declined",
    "not approved: nobody answered",
    "is blocked in unattended runs",
    "not configured on the server",
)


def blocked_by_a_refusal(steps: list[dict]) -> str:
    """The refusal the run reported, if its last tool result was one."""
    for step in reversed(steps or []):
        result = (step.get("result") or "").lower()
        for marker in BLOCKED_MARKERS:
            if marker in result:
                return step.get("tool", "a tool")
    return ""


def review(goal: str, steps: list[dict], final_text: str, call_ai_fn) -> Verdict:
    """
    One critic pass. Never raises — a failing critic accepts and logs.

    steps: [{"tool": name, "result": text}, ...] in execution order.
    """
    if not (final_text or "").strip():
        return Verdict(ACCEPT)

    prompt = CRITIC_PROMPT.format(
        goal=(goal or "(no goal recorded)")[:2000],
        transcript=format_transcript(steps),
        final=final_text[:3000],
    )
    try:
        response = call_ai_fn(
            [{"role": "user", "content": prompt}],
            "You are a strict but fair reviewer. You answer only with a verdict.",
        )
    except Exception as e:
        logger.warning(f"Critic call failed, accepting: {e}")
        return Verdict(ACCEPT)

    if isinstance(response, str) and response.startswith("❌"):
        # Router-level failure (all providers down/cooling) — not a verdict.
        logger.warning(f"Critic got a provider error, accepting: {response[:120]}")
        return Verdict(ACCEPT)

    verdict = parse_verdict(response)
    if not verdict.accepted:
        blocked_tool = blocked_by_a_refusal(steps)
        if blocked_tool:
            logger.info(
                f"🔍 Critic said REVISE, but the transcript shows '{blocked_tool}' was "
                "refused — reporting a blocker is a complete outcome. Accepting."
            )
            return Verdict(ACCEPT, raw=verdict.raw)
    logger.info(f"🔍 Critic: {verdict.kind}" + (f" — {verdict.reason[:120]}" if verdict.reason else ""))
    return verdict


def revision_message(reason: str) -> str:
    """The turn pushed back into the agent loop after a REVISE."""
    return REVISION_TEMPLATE.format(reason=reason)


def unresolved_note(reason: str) -> str:
    """
    Appended to the delivered result when the round cap is hit.

    The work still ships — but the reader sees what the critic never got
    satisfied about, rather than the concern being silently dropped.
    """
    return (
        "\n\n---\n⚠️ _Delivered with an unresolved review note (revision limit "
        f"reached): {reason.strip()[:500]}_"
    )
