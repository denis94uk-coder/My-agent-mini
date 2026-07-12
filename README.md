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

## 🧠 Optional AI providers

The built-in router can also use the provider integrations already in the bot, including Gemini, Groq, xAI, Cerebras, SambaNova, Together, Mistral, Cohere, OpenRouter, HuggingFace, Merge, and NVIDIA. Add only the keys you choose to use; the bot never requires all of them.

> The default Pollinations route is keyless. Optional free API keys improve reliability, but every provider still has its own quota and terms.

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

The default `POLLINATIONS_ENABLED=true` route needs no AI API key. It is best-effort only and can be rate-limited or unavailable. For stronger reliability, add one or more optional provider keys in `.env` (for example NVIDIA, Gemini, Groq, or Mistral). Save with **Ctrl+O**, Enter, then **Ctrl+X**.

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
├── requirements.txt    # Python dependencies (3 packages)
├── setup.sh            # One-click server setup
├── .env.example        # Template for API keys
├── .gitignore          # Protects .env
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
