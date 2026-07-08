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
) -> str:
    """
    Run the ReAct agent loop.

    Args:
        messages: Conversation history
        call_ai_fn: Function to call the AI (takes messages list, returns string)
        system_prompt: Base system prompt
        user_id: User ID for memory/facts

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

    for iteration in range(MAX_ITERATIONS):
        logger.info(f"🔄 Agent loop iteration {iteration + 1}/{MAX_ITERATIONS}")

        # Call AI with current messages
        response = call_ai_fn(working_messages, full_prompt)

        # Check for tool call
        tool_call = parse_tool_call(response)

        if tool_call is None:
            # No tool call — this is the final answer
            logger.info("✅ Agent gave final answer")
            return response

        # Execute the tool
        tool_name = tool_call["tool"]
        tool_args = tool_call["args"]

        # Pass user_id to remember tool
        if tool_name == "remember":
            tool_args["user_id"] = user_id

        logger.info(f"🔧 Using tool: {tool_name}({json.dumps(tool_args)[:100]})")
        tool_result = tools.run_tool(tool_name, tool_args)
        logger.info(f"📋 Tool result: {tool_result[:200]}...")

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
