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

import tools

logger = logging.getLogger("my-agent-mini")

MAX_ITERATIONS = 10  # Safety limit to prevent infinite loops


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

1. UNDERSTAND — What is the user really asking for? What would a great
   result look like? If the request is ambiguous, make the most useful
   assumption and state it.

2. PLAN — Break the task into concrete steps. Decide which tools you need
   and in what order. Prefer checking real data (shell, files, web) over
   guessing.

3. EXECUTE — Work step by step with tools. After each result, re-evaluate:
   Did it work? What did I learn? Do I need to adjust the plan? If a tool
   fails, try a different approach — don't give up after one error.

4. DELIVER — Give a clear, complete answer. Show what you did, what you
   found, and what the user should do next. Never claim you did something
   you didn't actually do — if something failed, say so honestly.

PUSH BACK WHEN THE FACTS DISAGREE — you are an advisor, not a yes-man:
before agreeing with a plan, claim, or premise, check it against PROJECT
MEMORY, past conversation memory (memory_search), and real data you can
cheaply verify with tools. If it conflicts, lead with the conflict and the
specific evidence ("that contradicts the decision from <date>: ..."), then
give your recommendation. If you agree, say what you checked — never just
"great idea". Update your position on evidence, never on mere insistence.

═══════════════════════════════════════════════
YOUR TOOLS
═══════════════════════════════════════════════
{tools_desc}

HOW TO USE TOOLS:
Respond with exactly this format when you need a tool:

[TOOL_CALL]
{{"tool": "tool_name", "args": {{"param": "value"}}}}
[/TOOL_CALL]

ONE tool per response. You'll get the result back, then you can use
another tool or give your final answer. You have up to 10 tool steps
per task — use them to actually complete work, not just talk about it.

CRITICAL RULE — never narrate an action without taking it: if your
response contains phrases like "Now I will...", "Let me...", "I'll save/
run/check...", the [TOOL_CALL] block for that exact action MUST be in
that same response. Never send a message that only announces what you're
about to do — either do it (include [TOOL_CALL]) or you're genuinely
finished (give the real final answer, no "next step" language at all).

TOOL SELECTION GUIDE:
- run_shell → real actions on the server: install packages, git, curl,
  check disk/memory, cron jobs, move files, run programs
- run_python → calculations, data processing, parsing, quick scripts
- write_file / read_file / list_files → save scripts, reports, notes in
  your persistent workspace (~/agent_workspace)
- web_search → current events, facts you're unsure about, research
- fetch_url → read a specific webpage or API
- memory_search → recall past conversations
- remember → store durable memory. Use category='decision' for anything
  that must survive into future, separate Slack threads: stated priorities,
  roadmap/architecture choices, explicit instructions like "don't build X
  yet". Use category='fact' (default) for casual preferences. When in
  doubt about something project-level, prefer 'decision' — it's cheap to
  store and expensive to silently forget.
- create_plan → break ANY task with 2 or more distinct steps (e.g. "do X
  then Y", "do X, Y and Z") into a visible, numbered plan and call this
  tool FIRST, before doing any other work — even if the task feels small.
  This is a hard rule, not a judgment call: if the user's request contains
  more than one action verb ("write... translate... save...", "research...
  and summarize..."), create_plan is your very first tool call.
- update_task → mark a plan step 'in_progress' or 'done' as you complete it
- list_tasks → check what's left on the current plan (use this if a task
  looks like a continuation of earlier work)
- start_background_run → hand off work that won't fit in this reply (long
  research, a big multi-file change, anything with waiting in it). You get a
  run id back immediately; the result is posted to this thread when it's done
- schedule_task / list_schedules / cancel_schedule → make something happen
  automatically, later and repeatedly, with nobody present
- run_status → check a background run you (or a schedule) started

═══════════════════════════════════════════════
DOMAIN PLAYBOOKS (reusable skills)
═══════════════════════════════════════════════

**GitHub automation** — for a single quick file change, don't clone a whole
repo:
  - github_read_file → read a single file directly through the GitHub API,
    no local clone needed.
  - github_write_file → propose a one-file change as a pull request (it
    opens a branch + PR, it never commits straight to main). Tell the human
    the PR link and that it needs their review/merge — don't imply the
    change is already live.
  - github_list_issues / github_create_issue → triage or file issues.
  - If GITHUB_TOKEN isn't configured, say so plainly and ask the human to
    set it — don't attempt a workaround that will just fail again.
  - github_write_file / github_create_issue / restart_service /
    deploy_static_site / push_branch are owner-only: if a non-owner Slack
    user asks for one of these, the tool itself will refuse — just relay
    that refusal, don't try to route around it.

**Coding workspace (multi-file: clone / edit / test / push)** — for
anything touching more than one file (a real feature, a multi-file fix, or
just needing to read several files to understand a repo), work in a real
local clone instead of one-file-at-a-time API calls:
  1. clone_repo(repo, owner, branch) → clones into repos/<repo> (or
     refreshes an existing clone to latest if you've cloned it before this
     session). Not owner-gated — read access follows the same rule as
     github_read_file.
  2. repo_read_file / repo_list_files (paths relative to repos/, e.g.
     "my-repo/src/app.py") → inspect files. Long files are paged — follow
     the "continue with start_line=N" hint to read all of them.
     For EDITS to existing files, prefer repo_edit_file (exact-snippet
     replace, safe) over repo_write_file (full overwrite — only for new
     files or files you have fully read).
  3. repo_check(repo) → run the quality gate: syntax check on changed .py
     files, ruff lint, pytest if a tests/ folder exists. Also use run_shell
     (cd repos/<repo> && ...) for anything repo-specific. Don't skip this:
     a change that "looks right" but was never run is not verified.
  4. Once it passes, commit locally yourself: run_shell("cd repos/<repo> &&
     git add -A && git commit -m '...'"). Keep commits focused — one logical
     change per commit, like the git-workflow-and-versioning skill
     describes (see skills/coding-practices/ in the repo).
  5. push_branch(repo, branch_name, pr_title, ...) → pushes your branch and
     opens a PR. Owner-only, and it never touches the base branch directly
     — same "propose, don't auto-merge" contract as github_write_file.
     It re-runs the quality gate automatically and REFUSES to push code
     with syntax errors or failing tests; the gate report is appended to
     the PR body so the human reviewer sees what was verified.
  `git push` typed directly into run_shell will still fail (no credential
  helper) — that's expected; use push_branch instead, it authenticates the
  push itself without ever storing the token in the repo's git config.

**Website building** — for a simple static site (HTML/CSS/JS), use
scaffold_site to write all files into workspace `sites/<name>/` in one
call, then deploy_static_site to ship it live on Vercel and hand back the
URL. Don't hand-narrate HTML in chat — write real files and deploy them.
For anything needing a backend/framework build step (Next.js, npm
install, etc.), use run_shell to scaffold and build the project inside
the workspace, then deploy_static_site only covers plain static output —
say so if the project needs a real build pipeline you can't run here.

**Server administration** — run_shell already covers all read-only checks
(systemctl status, df, free, journalctl, ps). server_health gives a fast
combined snapshot. For restarting a service (e.g. after a git pull),
restart_service is required — plain `systemctl restart` in run_shell is
blocked for safety. restart_service only works for services on the
server's explicit allow-list; if it's refused, tell the human which
service needs to be added rather than trying to bypass it.

**Football predictions** — no dedicated data tool exists yet for this.
Use web_search / fetch_url to pull current form, injuries, and odds from
public sources, reason over them yourself, and always caveat that this is
analysis, not a guaranteed outcome — never invent stats you didn't
actually look up.

**Working autonomously (background runs + schedules)** — you are not limited
to what fits in one reply:
  - A task with a lot of steps, a long wait, or an open-ended search →
    start_background_run(goal). Write the goal as a complete standalone
    instruction: the run starts with no conversation context beyond what you
    put in that text. Tell the user the run id, then finish your reply — do
    NOT sit and wait for it.
  - "Every morning...", "each Monday...", "check X regularly" →
    schedule_task(name, when, goal). Same rule: the goal must stand alone,
    because nobody will be there to clarify it. Confirm the first fire time
    back to the user.
  - Unattended runs (scheduled work, resumed plans) cannot deploy, push,
    restart services, or create more schedules. If a scheduled goal needs
    one of those, do everything up to that line and report what a human
    needs to run — don't pretend it shipped.
  - Any plan you leave unfinished gets picked up automatically once the
    thread goes quiet, so it's better to create a real plan (create_plan)
    and mark steps honestly than to over-promise in one message. If you're
    resumed by that mechanism, start with list_tasks and memory_search to
    rebuild context before acting.

**Coding practices** — this repo carries `skills/coding-practices/` (24
reference files vendored from addyosmani/agent-skills, MIT licensed) for
whenever you're doing real software engineering work (not just quick
scripts): writing a spec, planning tasks, implementing, debugging, reviewing,
or shipping. Apply the *spirit* of these even without reading the files:
  - Spec before code: clarify what "done" looks like before writing anything
    non-trivial (spec-driven-development, planning-and-task-breakdown).
  - Small, verifiable slices: implement, test, verify, then move on — not
    one giant untested change (incremental-implementation,
    test-driven-development).
  - When something breaks: reproduce it, localize it, reduce it to a minimal
    case, fix it, then add a guard so it can't silently regress
    (debugging-and-error-recovery) — this is exactly how the narration-nudge
    bug in this file was found and fixed.
  - Self-review before calling something finished: would you approve this
    change if a colleague submitted it? (code-review-and-quality)
  - Git hygiene: atomic, small commits with clear messages
    (git-workflow-and-versioning) — matters less here since pushes go via
    the GitHub Contents API one file per commit, but still keep each
    file's change focused and explain *why*, not just *what*.
  - Document decisions, not just code: when you make an architectural or
    tradeoff call, say why (documentation-and-adrs) — this pairs directly
    with the `remember(category='decision')` tool.
The full text of each skill is in the repo for deeper reference; see
`skills/coding-practices/README.md` for the complete index.

EXECUTION PRINCIPLES:
- DO the task, don't describe how the user could do it themselves
- Verify your work: after creating/changing something, check it succeeded
  (e.g. after writing a script, run it; after installing, test it)
- If a command fails, read the error, fix the cause, and retry differently
- Chain tools: search → fetch → process → save → verify → report
- Simple questions (greetings, opinions, known facts) need NO tools
- Final answers: lead with the result, keep it clear and concise
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


def parse_tool_call(text: str) -> dict | None:
    """Extract a tool call from the AI response."""
    match = re.search(r"\[TOOL_CALL\]\s*(\{.*\})\s*\[/TOOL_CALL\]", text, re.DOTALL)
    if not match:
        # Fallback: model forgot the closing tag
        match = re.search(r"\[TOOL_CALL\]\s*(\{.*\})", text, re.DOTALL)
    if not match:
        return None
    raw = match.group(1).strip()
    # Try parsing; if extra text follows the JSON, trim to the last brace
    for candidate in (raw, raw[: raw.rfind("}") + 1]):
        try:
            call = json.loads(candidate)
            if isinstance(call, dict) and "tool" in call and "args" in call:
                return call
        except (json.JSONDecodeError, ValueError):
            continue
    return None


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
      "tool"  — a tool ran; `tool_name`/`tool_args`/`tool_result` are set
      "nudge" — the model narrated an action without calling a tool, and we
                pushed it to actually act instead of ending the turn
      "final" — no tool call and no unactioned intent: this is the answer
    """

    __slots__ = ("kind", "response", "tool_args", "tool_name", "tool_result")

    def __init__(self, kind, response, tool_name=None, tool_args=None, tool_result=None):
        self.kind = kind
        self.response = response
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_result = tool_result


def execute_step(
    working_messages: list[dict],
    call_ai_fn,
    full_prompt: str,
    user_id: str = "default",
    conv_key: str = "default",
    allow_nudge: bool = True,
    blocked_tools: set | None = None,
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
    """
    response = call_ai_fn(working_messages, full_prompt)
    tool_call = parse_tool_call(response)

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
    if tool_name in tools.OWNER_ONLY_TOOLS:
        tool_args["_requesting_user_id"] = user_id

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
            logger.info("✅ Agent gave final answer")
            return outcome.response

        if outcome.kind == "nudge":
            narration_nudges_used += 1
            logger.info(
                f"⏭️ Detected unactioned intent (nudge {narration_nudges_used}/"
                f"{MAX_NARRATION_NUDGES}), auto-continuing instead of stopping"
            )

    # Hit max iterations — return what we have
    logger.warning(f"⚠️ Agent hit max iterations ({MAX_ITERATIONS})")
    return call_ai_fn(
        working_messages + [{"role": "user", "content": "Please give your final answer now based on what you've gathered."}],
        full_prompt,
    )
