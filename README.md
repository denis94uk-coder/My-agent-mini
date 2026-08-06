# 🤖 My-Agent-Mini

**A lightweight, hybrid AI Slack bot with a built-in smart router.**

No Docker. No second gateway service. Just one Python bot that connects to Slack, uses a keyless best-effort route by default, and falls back across any optional provider keys you add.

## ✨ Features

- **Built-in smart router** — keyless best-effort route plus optional providers
- **Automatic fallback** — skips rate-limited/unhealthy routes with cooldowns
- **Slack Integration** — DMs, @mentions, slash commands (`/ask`, `/clear`, `/providers`)
- **Threaded Conversations** — Maintains context within Slack threads
- **Near-zero cost** — runs on a Google Cloud e2-micro (or any 1 GB always-free
  VM) with a keyless AI route; an optional paid route is tried last and capped
- **Lightweight** — ~50MB RAM, single Python process
- **Auto-restart** — Systemd service keeps it running 24/7
- **Domain skills** — GitHub automation, server administration, and static
  website building/deployment, built in as tools (see below)
- **Durable project memory** — decisions and completed-task summaries
  persist across separate Slack threads/conversations, not just within one
  thread (see below)
- **Autonomous execution** — background runs that survive restarts,
  schedules that start work with nobody present, and automatic resumption
  of unfinished plans (see below)

## 🧠 Memory model

Two different things are stored in `memory.db`, and they behave differently:

- **Thread history** — the last ~20 messages of a specific Slack thread,
  scoped to that thread only (`conv_key`). This is what makes replies in an
  ongoing thread feel continuous; it's invisible to other threads.
- **Project memory** (`remember` tool, `category='decision'`, plus
  auto-generated `task_summary` entries when a plan finishes) — durable
  facts that get injected into **every** conversation for that user,
  regardless of which thread it's in. This is what lets the bot recall a
  stated priority or a past decision even in a brand-new thread, the same
  way a human coworker would.

Plain `remember` calls (`category='fact'`, the default) are ambient
preferences and only the most recent ones are kept in context — they're
not meant to be load-bearing. If something needs to survive long-term
(a roadmap item, an explicit instruction, an architecture choice), it
should be stored as a `decision`.

## 🤖 Autonomy: runs, schedules, and self-resuming plans

Everything else in the bot starts with a human typing. These three
mechanisms let it work without that.

### Background runs

A **run** is a durable task. Every step is written to SQLite (`runs` and
`run_events`) *before* the next one starts, so a run can be cancelled,
inspected, and — after a crash or a `systemctl restart` — resumed exactly
where it stopped, with its context intact, instead of starting the task
over. Runs execute on background worker threads, so long work never blocks
the chat.

```
/runs              list recent runs
/runs 12           detail + result for run #12
/runs cancel 12    stop it at its next step boundary
```

The agent starts one itself via `start_background_run` when work won't fit
in a single reply.

### Schedules

`schedule_task` (owner only) makes something happen later, repeatedly, with
nobody present — this is what turns the bot from reactive to self-starting.

| Spec | Meaning |
|---|---|
| `every 15m`, `every 2h`, `every 1d` | fixed interval (minimum 1 minute) |
| `hourly` | on the hour |
| `daily 09:00` | every day at 09:00 |
| `weekly mon 08:15` | Mondays at 08:15 |
| `0 9 * * 1-5` | raw 5-field cron — weekdays at 09:00 |

Times are the **server's** local timezone. `/schedules` lists them,
`/schedules cancel <name>` removes one.

### Self-resuming plans

`create_plan` already stored multi-step plans, but nothing drove them
forward: if the agent stopped at step 3 of 7, those steps sat there until
someone spoke. Now a sweeper picks them up — but only once **both** the
plan and its Slack thread have been idle for `PLAN_STALE_SECONDS`
(default 10 min). That idle check is deliberate: it means the bot never
talks over someone who is mid-conversation or interrupts a plan that's
genuinely waiting on a human answer. Capped at `PLAN_MAX_RESUMES` per
conversation per day.

### Critic gate

The agent decides for itself when it's finished — which is exactly where
this bot has failed before (announcing it saved something it never saved).
So after a final answer, a separate AI pass re-reads the goal, the **tool
transcript**, and the proposed answer, and returns `ACCEPT` or `REVISE` +
a specific reason. A `REVISE` goes back into the loop as another turn.

It generalizes what `_run_quality_gate` already does for code — run an
independent check before accepting, refuse to ship what fails it — to tasks
with no test suite to run. The critic grades against tool results only; the
agent's own prose is never evidence for itself.

Three behaviours worth knowing, each deliberate:

- **Fails open.** An unparseable or erroring critic ACCEPTs (logged). A
  broken grader must not become a broken agent.
- **A reported blocker counts as done.** An unattended run that stopped at a
  blocked deploy and said so is finished — otherwise the gate would demand
  the impossible thing every round until the cap.
- **The cap ships the work.** After `CRITIC_MAX_ROUNDS`, the result is
  delivered with the unresolved critique appended, so the concern reaches
  the human instead of being silently dropped.

On for background runs, off for live chat by default (`CRITIC_INTERACTIVE`)
— a human reading a reply can push back themselves; nobody is reading an
unattended run. Each critic call counts against the run's step budget.

### Governor: risk tiers and human approval

The `OWNER_SLACK_ID` lock answers *"is the human asking right now the
owner?"* — which means nothing once a cron job is the caller. The governor
adds the middle option between "allowed" and "refused": **ask**.

Every tool has a risk tier — `read`, `write_local`, `external`. In an
unattended run:

| Tier | What happens |
|---|---|
| `read`, `write_local` | Runs. Asking permission to run `ls` just teaches people to approve without reading. |
| `external` (deploy, push, restart, schedule) | The run **pauses on disk** and asks in Slack. |
| Run-spawning tools | Always refused. Asking can't make a fork bomb safe. |

```
/approvals          what's waiting
/approve 3          run it, exactly as requested
/deny 3 wrong week  refuse it; the agent finishes without it and says so
```

Only the owner can decide. A schedule marked `allow_risky` skips the queue —
pre-authorised, so a nightly deploy doesn't ask every night.

Three properties are deliberate:

- **Expiry is a deny.** An unanswered request times out (24h default) and the
  run resumes with a refusal. Silence must never become consent.
- **Pausing is durable.** The request is a row in SQLite and the run's status
  is `awaiting_approval`. Restart the service and it's still waiting; approve
  it a day later and the *exact* call it asked for runs, not a re-derived one.
- **Unclassified tools are EXTERNAL.** Forgetting to classify a new deploy
  tool is far worse than one unnecessary approval. A test asserts every
  registered tool has a tier, so the default never fires in practice.

### Cost accounting

`/costs` shows AI calls per provider. Free routes are best-effort; a paid
backup is a real monthly budget, and an agent that starts its own work can
spend it unattended. Name the paid routes (`PAID_PROVIDERS`) and cap them per
day (`PAID_DAILY_LIMIT`) — past the cap they drop out of rotation while free
routes keep serving, so the bot degrades instead of stopping.

### Other limits on unattended work

- **Approvals off?** Then external tools are a flat no again — there is
  nobody to ask, so the run does everything up to that line and reports it.
- **Creating more autonomous work is always refused**, opt-in or not: a run
  that can queue runs is one bad reasoning step from a fork bomb of them.
- **Budgets** — per-run step and time caps (`RUN_MAX_STEPS`,
  `RUN_MAX_SECONDS`); on hitting one the run reports what it finished
  rather than vanishing. Plus a daily cap on self-started runs
  (`RUN_DAILY_LIMIT`) so a misfiring schedule can't spin. Runs *you* ask
  for are never blocked by that cap.
- **Every step is auditable** in `run_events`, including refusals.

See `.env.example` for all the knobs.

### Tests

```bash
pytest tests/ -q     # 153 tests, no Slack workspace or API keys needed
```

## 🛠️ Domain Skills

Beyond general chat, the agent has built-in "skills" — real tools for
recurring domains of work, described in its system prompt so it reaches for
them automatically:

| Skill | Tools | Requires |
|---|---|---|
| **GitHub automation** (single file) | `github_read_file`, `github_write_file` (opens a PR, never commits to main), `github_list_issues`, `github_create_issue` | `GITHUB_TOKEN` (+ optional `GITHUB_DEFAULT_OWNER`/`GITHUB_DEFAULT_REPO`) |
| **Coding workspace** (multi-file) | `clone_repo`, `repo_read_file`/`repo_write_file`/`repo_list_files`, `run_shell` (to test), `push_branch` (opens a PR, never commits to main) | `GITHUB_TOKEN` |
| **Server administration** | `server_health`, `restart_service` (allow-listed services only) | `ALLOWED_SERVICES` + passwordless `sudo systemctl restart <service>` for that exact unit |
| **Website building** | `scaffold_site` (writes static site files), `deploy_static_site` (ships them to Vercel) | `VERCEL_TOKEN` |

All degrade gracefully with a clear error if their token/config isn't set —
see `.env.example` for setup steps. Plain `git push` typed into `run_shell`
does **not** work on a fresh server (no credential helper); `github_write_file`
(single file) and `push_branch` (whole branch, after cloning with
`clone_repo`) are the reliable paths for proposing a change — both open a
PR for human review instead of committing straight to a base branch.

**Owner lock:** `github_write_file`, `github_create_issue`,
`restart_service`, and `deploy_static_site` only run for the Slack user ID
in `OWNER_SLACK_ID` — anyone else gets refused. Set this before making the
bot reachable by more than just you; without it, these tools fail open
(anyone can trigger them using your credentials).

### Coding practices reference

`skills/coding-practices/` holds 24 reference skill files (vendored from
[addyosmani/agent-skills](https://github.com/addyosmani/agent-skills), MIT)
covering spec-writing, TDD, debugging, code review, git workflow, and
shipping checklists. `agent.py`'s system prompt carries a condensed
one-line-per-skill summary so the model applies these habits on any real
engineering task; see `skills/coding-practices/README.md` for the full
index and text.

## 🧠 Optional AI providers

The built-in router can also use the provider integrations already in the bot, including Gemini, Groq, xAI, Cerebras, SambaNova, Together, Mistral, Cohere, OpenRouter, HuggingFace, Merge, and NVIDIA. Add only the keys you choose to use; the bot never requires all of them.

> The default Pollinations route is keyless. Optional free API keys improve reliability, but every provider still has its own quota and terms. Merge Gateway can be added as a paid-credit route using one `mg_...` key; it is not an unlimited/free provider.

## 🚀 Quick Start

### 1. Create a free VM

Reference deployment is a **Google Cloud e2-micro** (2 vCPU burst, 1 GB RAM,
always-free tier in `us-west1`, `us-central1`, or `us-east1`), Ubuntu 22.04.
Anything comparable works — Oracle's `VM.Standard.E2.1.Micro` is the same
class of box, and the setup below is identical on both.

The 1 GB ceiling is a real design constraint, not a footnote: it's why there
is no Docker, no gateway service, no vector database, and why background runs
default to 2 workers.

### 2. SSH into your server

```bash
gcloud compute ssh your-instance-name        # GCP
ssh -i your-key.key ubuntu@YOUR-SERVER-IP    # or plain SSH anywhere else
```

### 3. Run setup script

```bash
curl -sSL https://raw.githubusercontent.com/denis94uk-coder/my-agent-mini/main/setup.sh | bash
```

Or manually:
```bash
git clone https://github.com/denis94uk-coder/my-agent-mini.git
cd my-agent-mini
chmod +x setup.sh
./setup.sh
```

### 4. Configure the router (no AI key is required)

```bash
nano .env
```

The default `POLLINATIONS_ENABLED=true` route needs no AI API key and is tried first, so an old/expired API key cannot block it. It is best-effort only and can be rate-limited or unavailable. For stronger reliability, add one or more optional provider keys in `.env` (for example NVIDIA, Gemini, Groq, or Mistral). Save with **Ctrl+O**, Enter, then **Ctrl+X**.

### 5. Start the bot

```bash
sudo systemctl start my-agent
```

### 6. Check it's running

```bash
sudo systemctl status my-agent
sudo journalctl -u my-agent -f  # live logs
```

## 📋 Slash Commands

| Command | Description |
|---------|-------------|
| `/ask <question>` | Quick one-shot question |
| `/clear` | Reset conversation memory |
| `/providers` | Show routes, keyless/key-backed status, and cooldown health |
| `/runs` | List background runs (`/runs <id>`, `/runs cancel <id>`) |
| `/schedules` | List scheduled tasks (`/schedules cancel <name>`) |
| `/approvals` | What the agent is waiting on a human to authorise |
| `/approve <id>` / `/deny <id> <why>` | Decide a pending request (owner only) |
| `/costs` | AI calls per provider, paid routes tracked separately |
| `/status`, `/health` | Capabilities snapshot / deep operational health |

## 🔧 Management Commands

```bash
# Start / Stop / Restart
sudo systemctl start my-agent
sudo systemctl stop my-agent
sudo systemctl restart my-agent

# View logs
sudo journalctl -u my-agent -f

# Edit API keys (restart after)
nano ~/my-agent-mini/.env
sudo systemctl restart my-agent

# Update code
cd ~/my-agent-mini && git pull
sudo systemctl restart my-agent
```

## 📁 Project Structure

```
my-agent-mini/
├── bot.py              # Slack app: events, slash commands, provider router
├── agent.py            # ReAct loop + the execute_step primitive + prompt
├── runner.py           # Durable run engine (queue, workers, resume, budgets)
├── triggers.py         # Schedules + the stalled-plan sweeper
├── memory.py           # SQLite-backed durable + recent memory
├── concept_graph.py    # NetworkX entity/relationship layer over memory.db
├── tools.py            # Tool implementations
├── critic.py           # Critic gate: is "done" actually done?
├── governor.py         # Risk tiers, approval queue, cost accounting
├── tests/              # 153 tests — no Slack or API keys needed
├── requirements.txt    # Python dependencies
├── setup.sh            # One-click server setup
├── .env.example        # Template for API keys + autonomy settings
├── .gitignore          # Protects .env
├── skills/
│   └── coding-practices/  # 24 reference skill files (see README above)
└── README.md           # This file
```

## 🔗 Related

- **[My-Agent](https://github.com/denis94uk-coder/My-agent)** — Full version with Docker, LiteLLM, web dashboard, and 12 AI providers. Needs a much larger box (4 vCPU / 24 GB).

## 💡 How Hybrid Failover Works

```
User sends message in Slack
         ↓
 Built-in router chooses a healthy route
         ↓ fails or rate-limited?
 Cool that route down and try the next route
         ↓
 Send response back to Slack
```

Paid routes are always sorted **last**, whatever order they were registered
in, and drop out entirely once `PAID_DAILY_LIMIT` is reached — so a free route
is never skipped in favour of one that costs money.

If a route is rate-limited, returns an error, or times out, the bot automatically cools it down and tries the next healthy route. The router is best-effort: it cannot guarantee unlimited free access, bypass authentication, or make an unavailable service work.

## 📊 Resource Usage

- **RAM:** ~50 MB
- **CPU:** < 1% idle, brief spikes when processing
- **Disk:** < 100 MB total
- **Network:** Minimal (API calls only)

Comfortable on a 1 GB always-free instance (Google e2-micro, Oracle
E2.1.Micro). Background runs add roughly 10-20 MB per active worker while
they're executing; `RUN_WORKERS=2` is the default for that reason.
