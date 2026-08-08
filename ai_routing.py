"""
Tagging AI calls with the kind of work they do.

`call_ai_fn` is injected into agent.py, critic.py and runner.py rather than
imported, which is what keeps those modules free of Slack imports and lets the
tests drive them with a scripted AI. That injection is also why the critic
could not ask for a different route than the agent loop: everything downstream
sees one opaque two-argument callable.

This adapter adds the third argument without breaking the second. A production
`call_ai` accepts `task=`; a test's `ScriptedAI` takes exactly two positional
arguments and would raise if handed a keyword. So the capability is *inspected*
rather than tried-and-excepted — catching TypeError here would also swallow a
genuine TypeError raised inside the AI call itself, and turn a real bug into a
silent fallback.
"""

import inspect
import logging

logger = logging.getLogger("my-agent-mini")


def _accepts_task(fn) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        # Builtins and some callables have no introspectable signature. Assume
        # the narrow contract; the worst case is routing as before.
        return False

    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "task":
            return True
    return False


def for_task(call_ai_fn, task: str):
    """Wrap `call_ai_fn` so its calls are tagged as `task`.

    Returns the original function untouched when it cannot accept the tag, so
    a caller can use this unconditionally.
    """
    if call_ai_fn is None or not _accepts_task(call_ai_fn):
        return call_ai_fn

    def tagged(messages, system_prompt=None, **kwargs):
        kwargs.setdefault("task", task)
        return call_ai_fn(messages, system_prompt, **kwargs)

    return tagged
