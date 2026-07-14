"""
ReAct Agent Loop — turns the chatbot into an agent that can plan and use tools.

Flow:
1. User sends a message
2. AI decides if it needs tools or can answer directly
3. If tools needed: AI outputs [TOOL_CALL], bot executes, feeds result back
4. Repeat until AI gives a final answer (max 5 iterations)

Works with ANY LLM provider (Gemini, Groq, xAI, etc.) — no native function
calling required. Uses a simple text-based protocol.
"""

import re
import json
import logging

import tools

logger = logging.getLogger("my-agent-mini")

MAX_ITERATIONS = 10  # Safety limit to prevent infinite loops


def get_agent_system_prompt(base_prompt: str, user_facts: list[str] = None) -> str:
    """Build the full system prompt with tool descriptions."""
    tools_desc = tools.get_tools_description()

    facts_section = ""
    if user_facts:
        facts_list = "\n".join(f"  - {f}" for f in user_facts[:10])
        facts_section = f"""
KNOWN FACTS ABOUT THIS USER:
{facts_list}
Use these to personalize your responses.
"""

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
- remember → store important facts about the user
- create_plan → break ANY task with 2 or more distinct steps (e.g. "do X
  then Y", "do X, Y and Z") into a visible, numbered plan and call this
  tool FIRST, before doing any other work — even if the task feels small.
  This is a hard rule, not a judgment call: if the user's request contains
  more than one action verb ("write... translate... save...", "research...
  and summarize..."), create_plan is your very first tool call.
- update_task → mark a plan step 'in_progress' or 'done' as you complete it
- list_tasks → check what's left on the current plan (use this if a task
  looks like a continuation of earlier work)

═══════════════════════════════════════════════
DOMAIN PLAYBOOKS (reusable skills)
═══════════════════════════════════════════════

**GitHub automation** — `git push` inside run_shell will fail on this
server (no credential helper configured). Never fight it or ask the human
to paste a token. Instead:
  - github_read_file → read a single file directly through the GitHub API,
    no local clone needed.
  - github_write_file → propose a file change as a pull request (it opens
    a branch + PR, it never commits straight to main). Tell the human the
    PR link and that it needs their review/merge — don't imply the change
    is already live.
  - github_list_issues / github_create_issue → triage or file issues.
  - Use run_shell + git only for read-only inspection (git log, git diff,
    git status) in a repo already cloned in the workspace.
  - If GITHUB_TOKEN isn't configured, say so plainly and ask the human to
    set it — don't attempt a workaround that will just fail again.
  - github_write_file / github_create_issue / restart_service /
    deploy_static_site are owner-only: if a non-owner Slack user asks for
    one of these, the tool itself will refuse — just relay that refusal,
    don't try to route around it.

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
    r"\bi'll (now )?(save|write|create|run|fetch|search|check|update|do)\b",
    r"\bi am going to (save|write|create|run|fetch|search|check|update)\b",
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

        # Call AI with current messages
        response = call_ai_fn(working_messages, full_prompt)

        # Check for tool call
        tool_call = parse_tool_call(response)

        if tool_call is None:
            # The model didn't include a [TOOL_CALL] block. Usually that means
            # it's truly done. But sometimes it just *narrates* the next step
            # ("Now, I will save the file.") without acting — if we return
            # that as the final answer, the bot looks stuck and the user has
            # to say "ok" to nudge it along. Catch that case and auto-continue
            # instead of ending the turn.
            if (
                looks_like_unactioned_intent(response)
                and narration_nudges_used < MAX_NARRATION_NUDGES
                and iteration < MAX_ITERATIONS - 1
            ):
                narration_nudges_used += 1
                logger.info(
                    f"⏭️ Detected unactioned intent (nudge {narration_nudges_used}/"
                    f"{MAX_NARRATION_NUDGES}), auto-continuing instead of stopping"
                )
                working_messages.append({"role": "assistant", "content": response})
                working_messages.append({
                    "role": "user",
                    "content": (
                        "[SYSTEM NUDGE] You described an action but did not include a "
                        "[TOOL_CALL] block to actually perform it. Do not repeat the "
                        "description — immediately output the [TOOL_CALL] block for "
                        "that exact action now."
                    ),
                })
                continue

            # No tool call — this is the final answer
            logger.info("✅ Agent gave final answer")
            return response

        # Execute the tool
        tool_name = tool_call["tool"]
        tool_args = tool_call["args"]

        # Pass user_id / conv_key to memory + planner tools automatically —
        # the model only needs to specify the task-relevant args.
        if tool_name == "remember":
            tool_args["user_id"] = user_id
        if tool_name in ("create_plan", "update_task", "list_tasks"):
            tool_args["conv_key"] = conv_key
        # Owner-gated tools need to know who is actually asking, so a
        # non-owner can't get the model to run them on their behalf.
        if tool_name in tools.OWNER_ONLY_TOOLS:
            tool_args["_requesting_user_id"] = user_id

        logger.info(f"🔧 Using tool: {tool_name}({json.dumps(tool_args)[:100]})")
        tool_result = tools.run_tool(tool_name, tool_args)
        logger.info(f"📋 Tool result: {tool_result[:200]}...")

        # Let the caller surface plan creation/updates live in Slack instead
        # of them only being visible inside the model's own reasoning.
        if on_tool_call and tool_name in ("create_plan", "update_task"):
            try:
                on_tool_call(tool_name, tool_result)
            except Exception:
                logger.warning("on_tool_call callback failed", exc_info=True)

        # Get any text the AI said before/after the tool call
        preamble = extract_final_text(response)

        # Add the AI's response and tool result to the conversation
        working_messages.append({
            "role": "assistant",
            "content": response,
        })
        working_messages.append({
            "role": "user",
            "content": f"[TOOL_RESULT for {tool_name}]\n{tool_result}\n[/TOOL_RESULT]\n\nUse this result to answer the user's question. If you need more information, use another tool. Otherwise, give your final answer.",
        })

    # Hit max iterations — return what we have
    logger.warning(f"⚠️ Agent hit max iterations ({MAX_ITERATIONS})")
    return call_ai_fn(
        working_messages + [{"role": "user", "content": "Please give your final answer now based on what you've gathered."}],
        full_prompt,
    )
