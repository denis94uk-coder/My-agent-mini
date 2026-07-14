# 🤖 My-Agent-Mini

**A lightweight, hybrid AI Slack bot with a built-in smart router.**

No Docker. No second gateway service. Just one Python bot that connects to Slack, uses a keyless best-effort route by default, and falls back across any optional provider keys you add.

## ✨ Features

- **Built-in smart router** — keyless best-effort route plus optional providers
- **Automatic fallback** — skips rate-limited/unhealthy routes with cooldowns
- **Slack Integration** — DMs, @mentions, slash commands (`/ask`, `/clear`, `/providers`)
- **Threaded Conversations** — Maintains context within Slack threads
- **Zero Cost** — Runs on Oracle Cloud free tier + free AI APIs
- **Lightweight** — ~50MB RAM, single Python process
- **Auto-restart** — Systemd service keeps it running 24/7
- **Domain skills** — GitHub automation, server administration, and static
  website building/deployment, built in as tools (see below)
- **Durable project memory** — decisions and completed-task summaries
  persist across separate Slack threads/conversations, not just within one
  thread (see below)

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

**Owner lock:** `run_shell`, `run_python`, `github_write_file`,
`github_create_issue`, `restart_service`, `deploy_static_site`, and
`push_branch` only run for the Slack user ID in `OWNER_SLACK_ID` — anyone
else gets refused. Set `OWNER_SLACK_ID` before making the bot reachable by
more than just you. If it's left unset, owner-only tools fail **closed**
(disabled for everyone, including you) rather than open.

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

### 1. Create Oracle Cloud Free Instance

- Shape: `VM.Standard.E2.1.Micro` (1 OCPU, 1 GB) — Always Free
- Or: `VM.Standard.A1.Flex` (up to 4 OCPU, 24 GB) — Always Free (if available)
- OS: Ubuntu 22.04

### 2. SSH into your server

```bash
ssh -i your-key.key ubuntu@YOUR-SERVER-IP
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
├── bot.py              # Main bot (single file, ~300 lines)
├── agent.py            # ReAct agent loop + system prompt
├── memory.py           # SQLite-backed durable + recent memory
├── tools.py            # Tool implementations
├── requirements.txt    # Python dependencies (3 packages)
├── setup.sh            # One-click server setup
├── .env.example        # Template for API keys
├── .gitignore          # Protects .env
├── skills/
│   └── coding-practices/  # 24 reference skill files (see README above)
└── README.md           # This file
```

## 🔗 Related

- **[My-Agent](https://github.com/denis94uk-coder/My-agent)** — Full version with Docker, LiteLLM, web dashboard, and 12 AI providers. Needs 4 OCPU / 24 GB (Oracle A1.Flex).

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

If a route is rate-limited, returns an error, or times out, the bot automatically cools it down and tries the next healthy route. The router is best-effort: it cannot guarantee unlimited free access, bypass authentication, or make an unavailable service work.

## 📊 Resource Usage

- **RAM:** ~50 MB
- **CPU:** < 1% idle, brief spikes when processing
- **Disk:** < 100 MB total
- **Network:** Minimal (API calls only)

Perfect for Oracle's E2.1.Micro (1 GB RAM) free instance!
