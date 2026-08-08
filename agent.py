"""
ReAct Agent Loop — turns the chatbot into an agent that can plan and use tools.

Flow:
1. User sends a message
2. AI decides if it needs tools or can answer directly
3. If tools needed: AI outputs [TOOL_CALL], bot executes, feeds result back
4. Repeat until AI gives a final answer (max 5 iterations)

Works with ANY LLM provider (Gemini, Groq, xAI, etc.) — no native function
calling required. Uses a simple text-based protocol.

The loop is built from `execute_step`, one AI call + at most one tool run.
`run_agent_loop` drives it synchronously for interactive Slack replies;
`runner.py` drives the same primitive one step at a time, persisting each
step to SQLite so an autonomous run can survive a restart and resume.
"""

import re
import json
import logging

import critic
import governor
import tools

logger = logging.getLogger("my-agent-mini")

MAX_ITERATIONS = 10  # Safety limit to prevent infinite loops
# Prose left in the last response after stripping its tool block, above which
# it can stand as the answer instead of buying another call to rephrase it.
MIN_SALVAGEABLE_ANSWER_CHARS = 200


def get_agent_system_prompt(base_prompt: str, user_facts: dict[str, list[str]] | None = None) -> str:
    """Build the full system prompt with tool descriptions."""
    tools_desc = tools.get_tools_description()

    user_facts = user_facts or {"durable": [], "recent": []}
    facts_section = ""
    durable, recent = user_facts.get("durable") or [], user_facts.get("recent") or []
    if durable or recent:
        parts = []
        if durable:
            # Decisions/task summaries: this is durable project memory —
            # render ALL of it (already capped in memory.get_facts), never
            # truncated further, so it survives across separate Slack
            # threads/conversations the way a real project memory should.
            parts.append(
                "PROJECT MEMORY (decisions, roadmap, completed work — true "
                "across ALL conversations, not just this thread):\n"
                + "\n".join(f"  - {f}" for f in durable)
            )
        if recent:
            parts.append(
                "RECENT NOTES ABOUT THIS USER:\n" + "\n".join(f"  - {f}" for f in recent)
            )
        facts_section = "\n\n" + "\n\n".join(parts) + (
            "\nTreat PROJECT MEMORY as ground truth: check new requests against it "
            "(e.g. does this contradict a stated priority or an earlier decision?) "
            "and say so instead of silently going along with something that conflicts. "
            "Use these to personalize your responses.\n"
        )

    return f"""{base_prompt}

{facts_section}
═══════════════════════════════════════════════
YOUR REASONING FRAMEWORK (how a full agent thinks)
═══════════════════════════════════════════════

For every non-trivial request, work in four phases:

1. UNDERSTAND — what is actually being asked, and what would a great result
   look like? If ambiguous, make the most useful assumption and state it.
2. PLAN — break it into concrete steps and pick the tools. Prefer checking
   real data (shell, files, web) over guessing.
3. EXECUTE — one step at a time. After each result: did it work, what did I
   learn, does the plan change? If a tool fails, try a different approach.
4. DELIVER — show what you did and what you found. Never claim you did
   something you didn't — if it failed, say so.

Before agreeing with any plan, claim, or premise, check it against PROJECT
MEMORY, past conversations (memory_search), and data you can verify with
tools. Lead with the conflict and the evidence when it disagrees.

═══════════════════════════════════════════════
YOUR TOOLS
═══════════════════════════════════════════════
{tools_desc}

HOW TO USE TOOLS:
Respond with exactly this format when you need a tool:

[TOOL_CALL]
{{"tool": "tool_name", "args": {{"param": "value"}}}}
[/TOOL_CALL]

ONE tool per response, and you get the result back before choosing again.
You have up to 10 tool steps per task — use them to complete work, not to
talk about it.

ONE EXCEPTION — read-only tools batch. When you need several independent
reads (web_search, fetch_url, read_file, list_files, repo_read_file,
repo_list_files, github_read_file, github_list_issues, memory_search,
graph_recall, server_health, list_tasks, list_schedules, run_status), emit
every [TOOL_CALL] block in the SAME response and you get all the results
together. Only batch reads whose arguments you already know — if one read's
arguments depend on another's result, they are not independent, so issue
them in separate responses. Never batch a tool that writes, pushes, deploys,
schedules, or remembers.

CRITICAL RULE — never narrate an action without taking it: if your
response contains phrases like "Now I will...", "Let me...", "I'll save/
run/check...", the [TOOL_CALL] block for that exact action MUST be in
that same response. Never send a message that only announces what you're
about to do — either do it (include [TOOL_CALL]) or you're genuinely
finished (give the real final answer, no "next step" language at all).

TOOL SELECTION RULES (the registry above says what each tool does; these
are the calls that are easy to get wrong):
- create_plan is a hard rule, not a judgment call: if the request contains
  more than one action verb ("write... translate... save..."), it is your
  very first tool call, before any other work.
- Prefer repo_edit_file (exact-snippet replace) over repo_write_file (full
  overwrite) on any file you have not fully read.

═══════════════════════════════════════════════
DOMAIN PLAYBOOKS (reusable skills)
═══════════════════════════════════════════════

**GitHub automation** — for a single quick file change, don't clone a whole
repo:
  - github_read_file → read one file through the API, no clone needed.
  - github_write_file → propose a one-file change as a PR (it opens a
    branch + PR, never commits to main). Give the human the PR link and say
    it needs review — don't imply the change is already live.
  - github_list_issues / github_create_issue → triage or file issues.
  - If GITHUB_TOKEN isn't configured, say so plainly and ask the human to
    set it — don't attempt a workaround that will just fail again.
  - run_shell / run_python / github_write_file / github_create_issue /
    restart_service / push_branch / schedule_task /
    start_background_run are owner-only: the tool itself refuses a non-owner,
    so relay that refusal rather than routing around it. With no
    OWNER_SLACK_ID configured they refuse everyone, owner included — say so
    plainly rather than retrying.

**Coding workspace (multi-file: clone / edit / test / push)** — for
anything touching more than one file (a real feature, a multi-file fix, or
just needing to read several files to understand a repo), work in a real
local clone instead of one-file-at-a-time API calls:
  1. clone_repo → clones into repos/<repo>, or refreshes an existing clone.
  2. repo_read_file / repo_list_files, paths relative to repos/ (e.g.
     "my-repo/src/app.py"). Long files page — follow the "continue with
     start_line=N" hint until you have the whole file.
  3. repo_check → the quality gate: syntax, ruff, pytest if tests/ exists.
     Don't skip it; a change that was never run is not verified.
  4. Commit locally: run_shell("cd repos/<repo> && git add -A && git commit
     -m '...'"), one logical change per commit.
  5. push_branch → pushes and opens a PR. It re-runs the gate and REFUSES
     to push syntax errors or failing tests, attaching the report to the PR.
  `git push` inside run_shell fails (no credential helper) — that's
  expected, use push_branch, which authenticates without storing a token.

**Website building** — for a static site, scaffold_site writes all files
into workspace `sites/<name>/` in one call. Write real files, don't narrate
HTML in chat. There is no publish tool: report the workspace path and let the
human ship it, and say plainly that you cannot deploy it yourself.

**Server administration** — run_shell covers read-only checks (systemctl
status, df, free, journalctl, ps); server_health is a fast combined
snapshot. Restarts require restart_service — plain `systemctl restart` in
run_shell is blocked, and restart_service only accepts services on the
server's allow-list. If it refuses, tell the human which service needs
adding rather than bypassing it.

**Working autonomously (background runs + schedules)** — you are not limited
to what fits in one reply:
  - Many steps, a long wait, or an open-ended search → start_background_run.
    "Every morning...", "each Monday..." → schedule_task. For both, the goal
    must be a complete standalone instruction — the run starts with no
    conversation context beyond that text, and nobody is there to clarify.
    Report the run id or first fire time, then finish your reply; never sit
    and wait for it.
  - Unattended runs cannot deploy, push, restart services, or create more
    schedules. If a scheduled goal needs one, do everything up to that line
    and report what a human must run — don't pretend it shipped.
  - An unfinished plan is resumed automatically once the thread goes quiet,
    so create_plan and honest step marking beat over-promising in one
    message. If you are the resumed run, start with list_tasks and
    memory_search to rebuild context before acting.

**Coding practices** — for real software engineering (not quick scripts):
spec before code, small verifiable slices, reproduce-localize-fix-guard when
debugging, self-review before calling it done, atomic commits explaining
*why*, and `remember(category='decision')` for architectural calls. Full
references live in `skills/coding-practices/` — read the relevant file when
a task warrants it; see its README.md for the index.

EXECUTION PRINCIPLES:
- DO the task, don't describe how the user could do it themselves
- Verify your work: after writing a script, run it; after installing, test it
- If a command fails, read the error, fix the cause, and retry differently
- Chain tools: search → fetch → process → save → verify → report
- Simple questions (greetings, opinions, known facts) need NO tools
- Be honest about limits: you cannot access private accounts, send emails,
  or act outside this server unless a tool allows it
"""


# Phrases that signal the model is *narrating an intended action* rather
# than giving an actual final answer. If a response matches one of these
# and contains no parseable [TOOL_CALL], the loop should nudge the model to
# continue instead of ending the turn — otherwise the bot posts "Now I will
# save the file." as its final Slack message and just... doesn't.
_INTENT_ONLY_PATTERNS = [
    r"\bi will now\b",
    r"\bnow,? i will\b",
    r"\bnow i('| a)?m going to\b",
    r"\blet me now\b",
    r"\bnext,? i will\b",
    # NOTE: "remember"/"note" must be in this verb list. Without it, a
    # response like "I'll remember it's 40 degrees Celsius" reads as a
    # final answer (no [TOOL_CALL] block ever gets emitted), so nothing
    # is actually persisted to memory.db — the model just *says* it will
    # remember something and then doesn't. Confirmed live: user told the
    # bot a fact, got this exact narration back, and it was gone in a new
    # thread. Keep any future save-intent verb added here too.
    r"\bi'll (now )?(save|write|create|run|fetch|search|check|update|do|remember|note)\b",
    r"\bi(?: will|'ll) (remember|note|keep in mind|store) that\b",
    r"\bi am going to (save|write|create|run|fetch|search|check|update|remember|note)\b",
]
_INTENT_ONLY_RE = re.compile("|".join(_INTENT_ONLY_PATTERNS), re.IGNORECASE)

MAX_NARRATION_NUDGES = 3  # cap extra nudges so a stuck model can't burn all iterations


def looks_like_unactioned_intent(text: str) -> bool:
    """True if the text describes an upcoming action but never actually calls a tool."""
    return bool(_INTENT_ONLY_RE.search(text))


def _parse_one_call(body: str) -> dict | None:
    """Parse the JSON inside a single [TOOL_CALL] block."""
    raw = body.strip()
    start, end = raw.find("{"), raw.rfind("}")
    candidates = [raw]
    if start >= 0 and end > start:
        candidates.append(raw[start:end + 1])
    for candidate in candidates:
        try:
            call = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(call, dict) and "tool" in call and "args" in call:
            return call
    return None


def parse_tool_calls(text: str) -> list[dict]:
    """
    Every tool call in the response, in order.

    Blocks are delimited by their tags rather than by brace matching, so an
    argument containing braces (a code snippet, a JSON payload) does not end
    the block early. A missing closing tag is tolerated: the block then runs
    to the next opening tag or to the end of the response.

    The single-call regex this replaces was greedy, so two blocks in one
    response matched from the first opening tag to the last closing tag and
    parsed as nothing at all. The response was then treated as a final
    answer — the model's narration was posted while neither tool ran.
    """
    calls = []
    for match in re.finditer(
        r"\[TOOL_CALL\](.*?)(?:\[/TOOL_CALL\]|(?=\[TOOL_CALL\])|\Z)", text, re.DOTALL
    ):
        call = _parse_one_call(match.group(1))
        if call is not None:
            calls.append(call)
    return calls


def parse_tool_call(text: str) -> dict | None:
    """The first tool call in the response, or None."""
    calls = parse_tool_calls(text)
    return calls[0] if calls else None


def extract_final_text(text: str) -> str:
    """Remove tool call blocks from text to get the human-readable part."""
    cleaned = re.sub(r"\[TOOL_CALL\].*?\[/TOOL_CALL\]", "", text, flags=re.DOTALL)
    # Also handle a tool call missing its closing tag
    cleaned = re.sub(r"\[TOOL_CALL\].*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


NUDGE_MESSAGE = (
    "[SYSTEM NUDGE] You described an action but did not include a "
    "[TOOL_CALL] block to actually perform it. Do not repeat the "
    "description — immediately output the [TOOL_CALL] block for "
    "that exact action now."
)


def tool_result_message(tool_name: str, tool_result: str) -> str:
    """The user-role turn we feed back after running a tool."""
    return (
        f"[TOOL_RESULT for {tool_name}]\n{tool_result}\n[/TOOL_RESULT]\n\n"
        "Use this result to answer the user's question. If you need more "
        "information, use another tool. Otherwise, give your final answer."
    )


class StepOutcome:
    """
    What happened in one agent iteration.

    kind:
      "tool"   — a tool ran; `tool_name`/`tool_args`/`tool_result` are set
      "nudge"  — the model narrated an action without calling a tool, and we
                 pushed it to actually act instead of ending the turn
      "final"  — no tool call and no unactioned intent: this is the answer
      "paused" — the tool needs human approval. It has NOT run and nothing was
                 appended to the transcript; the caller parks the run and
                 replays this exact call once a human decides.
    """

    __slots__ = ("kind", "response", "tool_args", "tool_name", "tool_result", "batch")

    def __init__(self, kind, response, tool_name=None, tool_args=None, tool_result=None):
        self.kind = kind
        self.response = response
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_result = tool_result
        # Per-tool detail when several read-only tools ran in one step. The
        # scalar fields above stay the combined view, so durability and replay
        # treat a batch exactly like any other step.
        self.batch = []


def _batch_label(tool_calls: list[dict]) -> str:
    """Name for a batched step, used in the transcript and in run events."""
    return " + ".join(dict.fromkeys(str(c.get("tool")) for c in tool_calls))


def _execute_read_batch(
    working_messages: list[dict],
    response: str,
    tool_calls: list[dict],
    user_id: str = "default",
    conv_key: str = "default",
    blocked_tools: set | None = None,
) -> StepOutcome:
    """
    Run several read-only tools from one response and feed back one result.

    The combined text is appended through `tool_result_message`, exactly as a
    single-tool step is, so a run that resumes from persisted events rebuilds
    the identical transcript rather than a re-shaped one.
    """
    sections, results = [], []
    for call in tool_calls:
        name = call["tool"]
        args = call["args"] if isinstance(call.get("args"), dict) else {}
        if name in ("list_tasks",):
            args["conv_key"] = conv_key
        for context_arg in tools.CONTEXT_TOOLS.get(name, ()):
            args[context_arg] = {"_user_id": user_id, "_conv_key": conv_key}[context_arg]
        if name in tools.owner_only_tools():
            args["_requesting_user_id"] = user_id

        if blocked_tools and name in blocked_tools:
            result = (
                f"❌ Blocked: '{name}' cannot run in an unattended background run."
            )
        else:
            result = tools.run_tool(name, args)
        logger.info(f"🔧 Batched tool: {name} → {result[:120]}...")
        sections.append(f"── {name} ──\n{result}")
        results.append({"tool": name, "args": args, "result": result})

    label = _batch_label(tool_calls)
    combined = "\n\n".join(sections)
    logger.info(f"📦 Ran {len(tool_calls)} read-only tools in one step: {label}")

    working_messages.append({"role": "assistant", "content": response})
    working_messages.append({"role": "user", "content": tool_result_message(label, combined)})

    outcome = StepOutcome("tool", response, label, {}, combined)
    outcome.batch = results
    return outcome


def execute_step(
    working_messages: list[dict],
    call_ai_fn,
    full_prompt: str,
    user_id: str = "default",
    conv_key: str = "default",
    allow_nudge: bool = True,
    blocked_tools: set | None = None,
    approval_fn=None,
    on_tool_call=None,
) -> StepOutcome:
    """
    Run exactly one iteration: call the AI, and execute a tool if it asked for one.

    Appends the resulting turns to `working_messages` in place (except for a
    final answer, which the caller decides what to do with). Callers that need
    durability persist the returned outcome after each call.

    blocked_tools: names refused before execution, with the refusal fed back as
        the tool result. Used for unattended runs, where nobody is watching to
        approve a deploy or a service restart.
    approval_fn: optional (tool_name, args) -> bool. False means the tool needs
        a human decision first: nothing runs, nothing is appended, and the
        caller gets a "paused" outcome to park on. Distinct from blocked_tools,
        which is a flat no the agent works around immediately.
    """
    response = call_ai_fn(working_messages, full_prompt)
    tool_calls = parse_tool_calls(response)

    # Several read-only tools in one response run as a batch. Each extra
    # round trip re-sends the whole system prompt, which is the dominant cost
    # per call and the reason free-tier token/minute limits bite: three files
    # read one at a time is three prompts, batched it is one.
    #
    # Only READ tier batches. A read is idempotent and side-effect-free, so
    # ordering and partial failure are harmless. Anything that writes runs one
    # per response, where the model sees each result before choosing again —
    # and EXTERNAL tools additionally need their own approval decision, which
    # a batch cannot express.
    if len(tool_calls) > 1 and all(
        governor.tier_of(c.get("tool")) == governor.READ for c in tool_calls
    ):
        return _execute_read_batch(
            working_messages, response, tool_calls,
            user_id=user_id, conv_key=conv_key, blocked_tools=blocked_tools,
        )

    tool_call = tool_calls[0] if tool_calls else None

    if tool_call is None:
        # The model didn't include a [TOOL_CALL] block. Usually that means
        # it's truly done. But sometimes it just *narrates* the next step
        # ("Now, I will save the file.") without acting — if we return that
        # as the final answer, the bot looks stuck and the user has to say
        # "ok" to nudge it along. Catch that case and auto-continue.
        if allow_nudge and looks_like_unactioned_intent(response):
            working_messages.append({"role": "assistant", "content": response})
            working_messages.append({"role": "user", "content": NUDGE_MESSAGE})
            return StepOutcome("nudge", response)
        return StepOutcome("final", response)

    tool_name = tool_call["tool"]
    tool_args = tool_call["args"] if isinstance(tool_call.get("args"), dict) else {}

    # Pass user_id / conv_key to memory + planner tools automatically —
    # the model only needs to specify the task-relevant args.
    if tool_name == "remember":
        tool_args["user_id"] = user_id
    if tool_name in ("create_plan", "update_task", "list_tasks"):
        tool_args["conv_key"] = conv_key
    if tool_name == "create_plan":
        tool_args["user_id"] = user_id
    # Autonomy tools report back to the conversation that started them.
    for context_arg in tools.CONTEXT_TOOLS.get(tool_name, ()):
        tool_args[context_arg] = {"_user_id": user_id, "_conv_key": conv_key}[context_arg]
    # Owner-gated tools need to know who is actually asking, so a
    # non-owner can't get the model to run them on their behalf.
    if tool_name in tools.owner_only_tools():
        tool_args["_requesting_user_id"] = user_id

    # Approval is checked before the block list: "ask a human" is a better
    # answer than "refused" when a human is reachable.
    if approval_fn is not None and not approval_fn(tool_name, tool_args):
        logger.info(f"🖐️ Tool {tool_name} needs approval — pausing instead of running it")
        return StepOutcome("paused", response, tool_name, tool_args)

    logger.info(f"🔧 Using tool: {tool_name}({json.dumps(tool_args, default=str)[:100]})")
    if blocked_tools and tool_name in blocked_tools:
        tool_result = (
            f"❌ Blocked: '{tool_name}' cannot run in an unattended background run — "
            "it changes state outside this server and nobody is watching to approve it. "
            "Do the rest of the work, then report that this step needs a human to run it."
        )
        logger.info(f"⛔ Blocked tool in unattended run: {tool_name}")
    else:
        tool_result = tools.run_tool(tool_name, tool_args)
    logger.info(f"📋 Tool result: {tool_result[:200]}...")

    # Let the caller surface plan creation/updates live in Slack instead
    # of them only being visible inside the model's own reasoning.
    if on_tool_call and tool_name in ("create_plan", "update_task"):
        try:
            on_tool_call(tool_name, tool_result)
        except Exception:
            logger.warning("on_tool_call callback failed", exc_info=True)

    working_messages.append({"role": "assistant", "content": response})
    working_messages.append({"role": "user", "content": tool_result_message(tool_name, tool_result)})

    return StepOutcome("tool", response, tool_name, tool_args, tool_result)


def run_agent_loop(
    messages: list[dict],
    call_ai_fn,
    system_prompt: str,
    user_id: str = "default",
    conv_key: str = "default",
    on_tool_call=None,
) -> str:
    """
    Run the ReAct agent loop.

    Args:
        messages: Conversation history
        call_ai_fn: Function to call the AI (takes messages list, returns string)
        system_prompt: Base system prompt
        user_id: User ID for memory/facts
        conv_key: Conversation key for the task planner (plan is per-thread)
        on_tool_call: Optional callback(tool_name, tool_result) fired right after a
            tool executes. Used so the caller (e.g. Slack bot) can surface the
            live plan (create_plan/update_task) as its own message instead of
            only showing it buried inside the final answer.

    Returns:
        Final response text
    """
    # Get user facts for personalization
    from memory import get_facts
    user_facts = get_facts(user_id)

    # Build full system prompt with tools
    full_prompt = get_agent_system_prompt(system_prompt, user_facts)

    # Working copy of messages for the loop
    working_messages = list(messages)
    narration_nudges_used = 0

    # For the critic gate: the goal it grades against, and the tool evidence
    # it grades with.
    goal_text = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )
    tool_trail = []
    critic_rounds = 0
    unresolved = ""

    for iteration in range(MAX_ITERATIONS):
        logger.info(f"🔄 Agent loop iteration {iteration + 1}/{MAX_ITERATIONS}")

        outcome = execute_step(
            working_messages,
            call_ai_fn,
            full_prompt,
            user_id=user_id,
            conv_key=conv_key,
            allow_nudge=(
                narration_nudges_used < MAX_NARRATION_NUDGES
                and iteration < MAX_ITERATIONS - 1
            ),
            on_tool_call=on_tool_call,
        )

        if outcome.kind == "final":
            # Critic gate (off by default here — see critic.interactive_enabled;
            # a human is reading this reply and can push back themselves).
            # Deliberately not skipped when no tool ran. An empty tool trail
            # looks like the cheapest call to save and is the opposite: an
            # answer claiming work with nothing behind it ("I've noted that
            # down", no remember call) is exactly what the critic exists to
            # catch, and it can only see that when it still runs.
            if (
                critic.interactive_enabled()
                and critic_rounds < critic.max_rounds()
                and iteration < MAX_ITERATIONS - 1
            ):
                verdict = critic.review(
                    goal_text, tool_trail, extract_final_text(outcome.response), call_ai_fn
                )
                if not verdict.accepted:
                    critic_rounds += 1
                    unresolved = verdict.reason
                    logger.info(f"🔁 Critic sent the answer back (round {critic_rounds})")
                    working_messages.append({"role": "assistant", "content": outcome.response})
                    working_messages.append({
                        "role": "user",
                        "content": critic.revision_message(verdict.reason),
                    })
                    continue
                unresolved = ""

            logger.info("✅ Agent gave final answer")
            if unresolved:
                return outcome.response + critic.unresolved_note(unresolved)
            return outcome.response

        if outcome.kind == "tool":
            # A batch grades as its individual tools — the critic reasons
            # about what was actually checked, not about a joined label.
            if outcome.batch:
                tool_trail.extend(
                    {"tool": r["tool"], "result": r["result"]} for r in outcome.batch
                )
            else:
                tool_trail.append({"tool": outcome.tool_name, "result": outcome.tool_result})

        if outcome.kind == "nudge":
            narration_nudges_used += 1
            logger.info(
                f"⏭️ Detected unactioned intent (nudge {narration_nudges_used}/"
                f"{MAX_NARRATION_NUDGES}), auto-continuing instead of stopping"
            )

    # Hit max iterations. Asking for a wrap-up costs another full call, and
    # the last response usually already carries usable prose alongside its
    # tool block — send that instead when there is enough of it to stand as
    # an answer, and only pay for the extra call when there is not.
    logger.warning(f"⚠️ Agent hit max iterations ({MAX_ITERATIONS})")
    salvaged = extract_final_text(outcome.response) if outcome.response else ""
    if len(salvaged) >= MIN_SALVAGEABLE_ANSWER_CHARS:
        logger.info("↩️ Using the last response as the answer instead of a wrap-up call")
        return salvaged
    return call_ai_fn(
        working_messages + [{"role": "user", "content": "Please give your final answer now based on what you've gathered."}],
        full_prompt,
    )
