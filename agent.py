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

MAX_ITERATIONS = 5  # Safety limit to prevent infinite loops


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
YOU HAVE ACCESS TO THESE TOOLS:
{tools_desc}

HOW TO USE TOOLS:
When you need to search the web, fetch a URL, run code, or search memory, respond with:

[TOOL_CALL]
{{"tool": "tool_name", "args": {{"param": "value"}}}}
[/TOOL_CALL]

You can use ONE tool per response. After receiving the result, you can use another
tool or give your final answer.

IMPORTANT RULES:
- Only use tools when genuinely needed — simple questions don't need tools
- After getting tool results, analyze them and respond to the user
- For web_search: provide a clear search query
- For fetch_url: provide a full URL starting with http:// or https://
- For run_python: write complete, runnable Python code
- For memory_search: use keywords from what you're looking for
- For remember: state the fact clearly and concisely
- When you have enough information, respond normally WITHOUT tool calls
- Always be helpful, clear, and concise in your final answer
"""


def parse_tool_call(text: str) -> dict | None:
    """Extract a tool call from the AI response."""
    match = re.search(r"\[TOOL_CALL\]\s*(\{.*?\})\s*\[/TOOL_CALL\]", text, re.DOTALL)
    if not match:
        return None
    try:
        call = json.loads(match.group(1))
        if "tool" in call and "args" in call:
            return call
    except json.JSONDecodeError:
        pass
    return None


def extract_final_text(text: str) -> str:
    """Remove tool call blocks from text to get the human-readable part."""
    cleaned = re.sub(r"\[TOOL_CALL\].*?\[/TOOL_CALL\]", "", text, flags=re.DOTALL)
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
