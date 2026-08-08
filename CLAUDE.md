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

Slack bot → autonomous agent. Single Python process, systemd, Ubuntu.

Deployment today: Google e2-micro free tier, 1 GB RAM. **Planned move to
self-hosted hardware.** Treat the 1 GB tuning (`RUN_WORKERS=2`,
`RUN_CONTEXT_LIMIT_CHARS`, the sub-agent deferral in the roadmap) as current
constraints, not permanent architecture — but don't pre-build for hardware
that doesn't exist yet either.

AI routes: keyless Pollinations first, merge.dev gateway as the paid backup
(sorted last, daily-capped). AI calls are not free — prefer deterministic
paths where they work. `CUSTOM_LLM_URL` already accepts any OpenAI-compatible
endpoint, which is the migration path to a local model when self-hosted.

No Docker, no gateway service. SQLite for everything.

## Architecture

| File | Role |
|---|---|
| `bot.py` | Slack events, slash commands, provider router (health + cooldowns), startup wiring |
| `agent.py` | `execute_step` (1 AI call + ≤1 tool) + `run_agent_loop` + system prompt |
| `runner.py` | Durable run engine: queue, workers, resume-after-crash, budgets |
| `triggers.py` | Schedules + stalled-plan sweeper |
| `critic.py` | Critic gate: re-reads goal + transcript, ACCEPT or REVISE |
| `governor.py` | Tool risk tiers, approval queue, cost accounting, rate-limit pacing |
| `memory.py` | Conversations, facts, plans, thread summaries, FTS5 |
| `concept_graph.py` | NetworkX entity/relation layer over `memory.db` |
| `tools.py` | All tool impls + registry + owner lock |
| `workflows.py` | Preset recurring jobs (goal text + schedule), started via `/workflow` |
| `tests/` | 277 tests, no Slack/keys/network needed |

Key invariant: **both** the interactive loop and the run engine drive
`agent.execute_step`. One implementation of the tool protocol. Don't fork it.

Tool protocol is text-based (`[TOOL_CALL]{json}[/TOOL_CALL]`), one tool per
response — works on any provider, no native function calling required.

## Rules that bit us before

- **Tokens per minute binds before tokens per day.** Groq free tier is 12k
  TPM against 100k TPD; Gemini pairs 250k TPM with ~10 *requests*/min. The
  loop fires up to 10 calls back to back, so a route dies mid-task with 95%
  of its daily budget unspent. `governor` paces on both ceilings before the
  call. Shrinking the prompt does not fix this — it only moves which step fails.
- **A 429 states its own reset window.** Read `Retry-After` /
  `x-ratelimit-reset-*`, don't guess. The old flat 90s was wrong both ways:
  it sat out 8-second token windows and under-waited daily exhaustion.
  `Retry-After` also permits an HTTP-date, whose digits parse as a plausible
  duration — handle date form first.
- **Only READ tools batch.** Several reads in one response cost one call
  instead of N. A write needs its result seen before the next choice, and an
  EXTERNAL tool needs its own approval decision, which one batched step
  cannot express. The batch persists as *one* `tool_result` event so resume
  rebuilds an identical transcript.
- **Never narrate an action without taking it.** `_INTENT_ONLY_PATTERNS` in
  `agent.py` catches "I'll remember that" with no tool call. Live bug: bot said
  it remembered, didn't. Add any new save-intent verb to that list.
- **Owner lock fails CLOSED.** No `OWNER_SLACK_ID` ⇒ every owner-only tool
  refuses, for everyone. It used to fail open, which handed shell access to any
  Slack user on a fresh install. `run_shell`/`run_python` are owner-only.
- **Owner-only and tier are different axes.** Owner-only = who may invoke.
  Tier = how far the effect reaches. Only one direction is an invariant:
  EXTERNAL ⇒ owner-only. `run_shell` is owner-only but WRITE_LOCAL, because
  gating `ls` behind Slack approval trains people to approve blind.
- **Unattended ≠ owner-approved.** `OWNER_SLACK_ID` asks "is the human asking
  right now the owner?" — meaningless when cron is the caller. An unattended
  run acts as the schedule's recorded owner, and what it may do is decided by
  *tier*, not by the owner-only list: `UNATTENDED_BLOCKED_TOOLS` is always a
  flat no, EXTERNAL parks for approval (or is a flat no with approvals off),
  and everything else runs. That deliberately includes owner-only WRITE_LOCAL
  tools — `repo-review` cannot run the test suite without `run_shell`. Gating
  those behind Slack approval would put an "approve `ls`?" prompt in front of
  a human, which is how an approval queue becomes a button people press blind.
- **Plan sweeper must not talk over a human.** Resume only when plan *and*
  Slack thread both idle ≥ `PLAN_STALE_SECONDS`. Don't drop that check.
- **Durability boundary is the step, never mid-tool.** Persist to `run_events`
  after a step completes; cancel/budget checks happen between steps.
- **Critic gate fails open, and a blocker is a valid outcome.** Unparseable
  verdict → ACCEPT. Honestly-reported blocked tool / missing token → ACCEPT.
  Drop either rule and the gate loops demanding impossible work. Round cap
  ships the result with the critique attached; it never eats the work.
- **Long runs need the fold.** `rebuild_messages` replays the whole
  transcript; past `RUN_CONTEXT_LIMIT_CHARS` it gets compacted to a progress
  note. The fold is persisted as a `compaction` event and resume rebuilds
  *from* it — otherwise recovery re-inflates the context the fold shrank.
  Summariser failure falls back to a deterministic digest, no AI call.
- **Schedules don't stack.** A schedule won't fire while its own previous run
  is active; `next_run` still advances so it can't get stuck due.
- **Retry ≠ resume.** `attempts`/`next_attempt_at` = the run errored (backoff,
  capped). `resume_count` = the process died mid-run. Config errors
  (`_PERMANENT_ERROR_MARKERS`) never retry.
- **Three levels unattended, not two.** Flat-blocked (`UNATTENDED_BLOCKED_TOOLS`,
  spawns more runs) → ask (EXTERNAL tier, parks the run) → allowed. `allow_risky`
  = pre-authorised, skips the queue. Approvals off = EXTERNAL back to flat no.
- **Expiry is a deny.** Unanswered approvals time out and unpark the run with a
  refusal. Silence never becomes consent.
- **A parked run holds no worker.** Status `awaiting_approval`, not claimable,
  and `recover_interrupted_runs` must leave it alone (it only touches `running`).
- **New tool ⇒ new tier.** `test_every_registered_tool_has_a_tier` fails the
  build otherwise. Unclassified defaults to EXTERNAL by design.
- **A hung run needs the watchdog.** Wall-clock budget is only checked between
  steps, so a run stuck inside a tool stays `running` forever and blocks its
  schedule. `sweep_stuck_runs` fails it on a stale heartbeat — it must never
  re-queue it, since the worker thread may still be live inside that tool.
- **Paid routes sort last.** Registration order in `build_providers` is just
  block order; the sort in `governor.is_paid` terms is what stops a free route
  being skipped for a paid one.
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
5. ✅ Governor (`governor.py`) — risk tiers, durable approve/deny queue with
   expiry-as-deny, per-provider cost accounting + paid daily cap
6. ⛔ Sub-agents / native function calling — assessed and declined, not
   deferred. Native tool-calling only works on the OpenAI-compatible routes,
   not keyless Pollinations, so it would fork the one tool protocol (see the
   key invariant above) to fix parse failures that aren't actually occurring.
   Sub-agents multiply AI calls and RAM on a 1 GB box with 2 workers, and cut
   against `UNATTENDED_BLOCKED_TOOLS`. Revisit only if the constraints change.

7. ✅ Quota-aware routing (`governor`) — per-route token *and* request minute
   windows checked before the call, `Retry-After` honoured, `ROUTER_ORDER` for
   explicit route preference, read-only tool batching
8. ✅ Workflows (`workflows.py`) — the recurring jobs the agent exists to do,
   as preset goals written for unattended runs: `repo-review`, `repo-health`,
   `ops-watch`, `decision-log`. Started with `/workflow start <name>`, which
   goes through the same `triggers.add_schedule` a human schedule uses — no
   second execution path

Still open, in rough value order: critic/summariser on a stronger route than
the agent itself; interactive loop handing off to a background run when it
hits `MAX_ITERATIONS`; vision (`images=`) in runs, currently interactive-only;
embeddings for memory retrieval (deliberately deferred — FTS5 fits 1 GB).
