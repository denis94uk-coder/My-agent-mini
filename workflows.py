"""
Workflows — the recurring jobs this agent is actually for.

Every capability here already existed; what was missing was a *use*. Tools,
schedules, runs and the critic gate were all in place, and the bot still sat
waiting to be asked something. A workflow names a job worth doing on its own,
writes the goal properly once, and schedules it.

Two constraints shape every goal below.

**They run unattended.** Nobody is present to clarify, so a goal must stand
alone: `runner` starts a run with no conversation context beyond this text.
Deploys, pushes, service restarts and new schedules are blocked in unattended
runs, so a goal that needs one must be written to stop and report rather than
to fail. Each goal below ends by saying what to do at that boundary.

**They run on a token budget.** A free-tier route affords roughly twenty AI
calls a day, and one agent loop can spend ten. So each goal states the order
to work in and when to stop, rather than inviting open-ended exploration —
and reads are batched, since several in one response cost one call rather
than one each. A workflow that quietly consumed the day's budget would take
the interactive bot down with it, which is the failure worth designing out.

Presets are text, not code paths: `start` hands the goal to the same
`triggers.add_schedule` a human would reach through `schedule_task`. There is
no second execution path to keep in step with the first.
"""

import logging

logger = logging.getLogger("my-agent-mini")


# `when` accepts what triggers.parse_spec accepts: 'every 30m', 'hourly',
# 'daily 09:00', 'weekly mon 09:00', or a 5-field cron expression.
PRESETS: dict[str, dict] = {
    "repo-review": {
        "when": "daily 09:00",
        "summary": "Review open PRs on the default repo and report what needs attention.",
        "needs": ["GITHUB_TOKEN", "GITHUB_DEFAULT_OWNER", "GITHUB_DEFAULT_REPO"],
        "goal": (
            "Review the open pull requests on the configured default GitHub repo "
            "and report their state.\n"
            "\n"
            "1. Call github_list_pull_requests to get the open PRs. If there are "
            "none, say exactly that and stop — do not go looking for other work.\n"
            "2. For each PR, up to five, call github_pr_status. Batch these calls: "
            "they are read-only, so issue them in one response rather than one per "
            "response.\n"
            "3. Report one line per PR: number, title, whether it is mergeable, and "
            "the CI result. Lead with any PR that has failing checks or a merge "
            "conflict, since those are the ones needing action.\n"
            "4. For a PR with failing CI, name the failing check and link it. Do "
            "not guess at the cause from the check name alone — say the logs were "
            "not read if they were not.\n"
            "\n"
            "Report only. Do not push commits, do not open or merge pull requests, "
            "and do not comment on GitHub — this run is unattended and those tools "
            "are blocked in it. If a PR needs a change, say which change and leave "
            "it to a human."
        ),
    },
    "repo-health": {
        "when": "weekly mon 09:00",
        "summary": "Clone the default repo, run the quality gate, and report what broke.",
        "needs": ["GITHUB_TOKEN", "GITHUB_DEFAULT_OWNER", "GITHUB_DEFAULT_REPO"],
        "goal": (
            "Check that the configured default GitHub repo still builds and passes "
            "its own tests.\n"
            "\n"
            "1. clone_repo to get a fresh copy of the default branch.\n"
            "2. repo_check to run the quality gate: syntax, ruff, and pytest.\n"
            "3. If everything passes, report one line saying so with the test count, "
            "and stop.\n"
            "4. If something fails, report the failing test or check by name with "
            "the relevant part of its output. Read the failing source file before "
            "explaining why it fails — an explanation invented from the test name "
            "is worse than reporting the failure alone.\n"
            "\n"
            "Do not fix anything. This run is unattended, so pushes are blocked, "
            "and a fix nobody reviewed is not worth the risk of being wrong about "
            "the cause. Report what a human should look at."
        ),
    },
    "ops-watch": {
        "when": "every 6h",
        "summary": "Check the server's disk, memory and services; report only real problems.",
        "needs": [],
        "goal": (
            "Check this server's health and report only what is actually wrong.\n"
            "\n"
            "1. Call server_health for the combined snapshot.\n"
            "2. Flag any of: disk above 85% full, available memory below 100 MB, "
            "the bot's own process not running, or a service in a failed state.\n"
            "3. If nothing crosses those lines, reply with exactly 'All clear' and "
            "the disk and memory figures on one line. Do not elaborate, and do not "
            "list what you checked — a clean check is one line.\n"
            "4. If something does cross a line, use run_shell to gather the "
            "specifics (what is consuming the disk, which service failed and its "
            "recent log lines) and report those with the numbers.\n"
            "\n"
            "Do not restart services or delete files. Restarts are blocked in an "
            "unattended run, and deleting something to reclaim disk is exactly the "
            "action that should have a human behind it. Say what needs running."
        ),
    },
    "decision-log": {
        "when": "weekly fri 17:00",
        "summary": "Summarise the week's decisions from memory into one durable note.",
        "needs": [],
        "goal": (
            "Summarise what was decided this week so it survives past the threads "
            "it was decided in.\n"
            "\n"
            "1. Use memory_search and graph_recall together — batch them, they are "
            "read-only — to find decisions, plans and completed work from the last "
            "seven days.\n"
            "2. Write a summary of at most ten bullets: what was decided, what "
            "shipped, and what was explicitly deferred and why. Deferrals matter "
            "as much as decisions; 'we chose not to build X yet' is the item most "
            "often lost.\n"
            "3. Save it with remember(category='decision') so it is durable across "
            "threads rather than living only in this run's output.\n"
            "4. Post the same summary as the run's reply.\n"
            "\n"
            "Include only decisions you actually found in memory. If the week was "
            "quiet, say so and save nothing — an invented summary of a quiet week "
            "poisons the memory that later runs read as ground truth."
        ),
    },
}


def describe() -> str:
    """Human-readable list of what is available, for Slack."""
    lines = ["*Available workflows:*\n"]
    for name, preset in PRESETS.items():
        needs = f" _(needs {', '.join(preset['needs'])})_" if preset["needs"] else ""
        lines.append(f"• *{name}* — {preset['summary']}\n  _{preset['when']}_{needs}")
    lines.append("\n_`/workflow start <name>` to schedule one._")
    return "\n".join(lines)


def missing_config(name: str) -> list[str]:
    """Environment variables a preset needs that are not set."""
    import os
    preset = PRESETS.get(name)
    if not preset:
        return []
    return [key for key in preset["needs"] if not os.getenv(key)]


def start(name: str, owner_user_id: str, channel: str = "", thread_ts: str = "",
          when: str = "") -> str:
    """
    Schedule a preset, through the same path a human-written schedule takes.

    Refuses when the preset's configuration is missing, because the failure
    would otherwise arrive as a scheduled run reporting a missing token at
    9am — later, quieter, and harder to connect back to this moment.
    """
    import triggers

    preset = PRESETS.get(name)
    if not preset:
        return f"❌ No workflow named '{name}'. Known: {', '.join(PRESETS)}"

    absent = missing_config(name)
    if absent:
        return (
            f"❌ '{name}' needs {', '.join(absent)} configured in .env first. "
            "Scheduling it now would just fail on its first run."
        )

    try:
        sched = triggers.add_schedule(
            name=name,
            spec=when or preset["when"],
            goal=preset["goal"],
            owner_user_id=owner_user_id,
            channel=channel,
            thread_ts=thread_ts,
        )
    except ValueError as e:
        return f"❌ {e}"

    import time
    next_at = time.strftime("%a %Y-%m-%d %H:%M", time.localtime(sched["next_run"]))
    logger.info(f"🗓️ Workflow '{name}' scheduled ({sched['spec']})")
    return (
        f"✅ Workflow *{name}* scheduled ({sched['spec']}) — first run {next_at} "
        f"(server time).\n_{preset['summary']}_\n"
        "Unattended runs cannot push, deploy or restart services; this one will "
        "report what needs a human instead."
    )
