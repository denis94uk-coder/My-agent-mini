# 🤖 My-Agent-Mini

**A lightweight, hybrid AI Slack bot that runs on any free server.**

No Docker. No database. No complex setup. Just a single Python script that connects to Slack and routes your messages through up to 10 free AI providers with automatic failover.

## ✨ Features

- **Hybrid AI** — Up to 10 free AI providers with automatic failover
- **Slack Integration** — DMs, @mentions, slash commands (`/ask`, `/clear`, `/providers`)
- **Threaded Conversations** — Maintains context within Slack threads
- **Zero Cost** — Runs on Oracle Cloud free tier + free AI APIs
- **Lightweight** — ~50MB RAM, single Python process
- **Auto-restart** — Systemd service keeps it running 24/7

## 🧠 Supported AI Providers (all free!)

| # | Provider | Free Tier | Speed |
|---|----------|-----------|-------|
| 1 | Google Gemini | 15 RPM, 1M tokens/day | ⚡ Fast |
| 2 | Groq | 30 RPM | ⚡⚡ Ultra-fast |
| 3 | xAI (Grok) | 60 RPM | ⚡ Fast |
| 4 | Cerebras | Free tier | ⚡⚡ Ultra-fast |
| 5 | SambaNova | Free tier | ⚡ Fast |
| 6 | Together AI | $5 free credit | ⚡ Fast |
| 7 | Mistral | Free tier | ⚡ Fast |
| 8 | Cohere | 20 RPM | ⚡ Fast |
| 9 | OpenRouter | Free models | ⚡ Fast |
| 10 | HuggingFace | Free tier | 🐢 Slower |

> Only need ONE provider to work. More providers = better reliability.

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

### 4. Add your API keys

```bash
nano .env
```

Add your Slack tokens and at least one AI provider API key.

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
| `/providers` | Show active AI providers |

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
   Try Provider #1 (Gemini)
         ↓ fails?
   Try Provider #2 (Groq)
         ↓ fails?
   Try Provider #3 (Grok)
         ↓ fails?
   ... continues through all configured providers
         ↓
   Send response back to Slack
```

If a provider is rate-limited, returns an error, or times out, the bot automatically tries the next one. You get seamless responses regardless of which provider handles them.

## 📊 Resource Usage

- **RAM:** ~50 MB
- **CPU:** < 1% idle, brief spikes when processing
- **Disk:** < 100 MB total
- **Network:** Minimal (API calls only)

Perfect for Oracle's E2.1.Micro (1 GB RAM) free instance!
