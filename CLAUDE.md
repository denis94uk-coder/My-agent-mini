# CLAUDE.md

## Operating manual applies to you

`operating_manual.md` in this repo is not just the bot's prompt — it governs
any agent working in this repo, including you. Read it before non-trivial
work. Load-bearing sections in practice: 4 (re-derive every number/claim,
including ones merely passing through), 5 (label inferences inline, at the
claim), 6 (attack your own conclusion before sending), 9 (push back with
evidence, not agreement), and the pre-send self-test.

## Response style — "caveman md"

Manual Section 10, applied to chat. Terse markdown. Save tokens on prose,
never on work. Target ~1/3 the words.

- short bullets, drop articles/filler ("push done" not "I have now pushed")
- no preamble, no recap of what user just said, no closing offers unless real fork
- lead with result
- full depth still applies: same research, same tests, same verification
- code, comments, commit messages, PR bodies, docs stay normal prose — this
  rule is about chat replies only
- compression removes words, never content: a correctness flag, an inference
  label, a surviving risk, or a disagreement each become one terse line —
  never zero lines
- terse ≠ vague. "needs GITHUB_TOKEN, `.env` has none" not "config issue"

## What this is

Slack bot → autonomous agent. Single Python process, systemd, Oracle free
tier (1 GB RAM). No Docker, no gateway service. SQLite for everything.

## Architecture

| File | Role |
|---|---|
| `bot.py` | Slack events, slash commands, provider router (health + cooldowns), startup wiring |
| `agent.py` | `execute_step` (1 AI call + ≤1 tool) + `run_agent_loop` + system prompt |
| `runner.py` | Durable run engine: queue, workers, resume-after-crash, budgets |
| `triggers.py` | Schedules + stalled-plan sweeper |
| `critic.py` | Critic gate: re-reads goal + transcript, ACCEPT or REVISE |
| `memory.py` | Conversations, facts, plans, thread summaries, FTS5 |
| `concept_graph.py` | NetworkX entity/relation layer over `memory.db` |
| `tools.py` | All tool impls + registry + owner lock |
| `tests/` | 103 tests, no Slack/keys/network needed |

Key invariant: **both** the interactive loop and the run engine drive
`agent.execute_step`. One implementation of the tool protocol. Don't fork it.

Tool protocol is text-based (`[TOOL_CALL]{json}[/TOOL_CALL]`), one tool per
response — works on any provider, no native function calling required.

## Rules that bit us before

- **Never narrate an action without taking it.** `_INTENT_ONLY_PATTERNS` in
  `agent.py` catches "I'll remember that" with no tool call. Live bug: bot said
  it remembered, didn't. Add any new save-intent verb to that list.
- **Unattended ≠ owner-approved.** `OWNER_SLACK_ID` asks "is the human asking
  right now the owner?" — meaningless when cron is the caller. Unattended runs
  block `OWNER_ONLY_TOOLS` (opt-in via schedule `allow_risky`) and always block
  `UNATTENDED_BLOCKED_TOOLS` (tools that spawn more autonomous work).
- **Plan sweeper must not talk over a human.** Resume only when plan *and*
  Slack thread both idle ≥ `PLAN_STALE_SECONDS`. Don't drop that check.
- **Durability boundary is the step, never mid-tool.** Persist to `run_events`
  after a step completes; cancel/budget checks happen between steps.
- **Critic gate fails open, and a blocker is a valid outcome.** Unparseable
  verdict → ACCEPT. Honestly-reported blocked tool / missing token → ACCEPT.
  Drop either rule and the gate loops demanding impossible work. Round cap
  ships the result with the critique attached; it never eats the work.
- `git push` in `run_shell` fails on the server (no credential helper). Use
  `push_branch` / `github_write_file` — both open a PR, never commit to main.

## Testing

```bash
pytest tests/ -q          # must stay green
ruff check .              # non-blocking; repo has pre-existing findings
```

Tests monkeypatch `memory.DB_PATH` to a tmpdir and inject a scripted AI —
keep new modules free of Slack imports so this stays possible.

## Roadmap (agreed phases)

1. ✅ Durable run engine
2. ✅ Triggers / scheduler
3. ✅ Plan executor (sweeper)
4. ✅ Critic gate (`critic.py`) — after "done", a critic pass re-reads goal +
   tool transcript, ACCEPTs or returns "not done because X". On for runs,
   opt-in for chat (`CRITIC_INTERACTIVE`), capped rounds
5. ⬜ Governor — tool risk tiers, Slack approve/deny queue for irreversible
   actions, cost/step accounting, full audit trail
6. ⬜ Optional: sub-agent delegation, native function calling where supported
